package com.example;

import java.util.*;
import java.io.*;
import org.json.*;
import org.ejml.simple.*;
import org.opencv.core.*;
import org.opencv.calib3d.*;
import org.opencv.imgproc.*;
import org.opencv.videoio.*;
import org.opencv.highgui.*;
import org.opencv.utils.*;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }

    public static Tuple<SimpleMatrix, SimpleMatrix> loadCameraModel(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            jsonBuilder.append(line);
        }
        reader.close();
        JSONObject jsonData = new JSONObject(jsonBuilder.toString());

        SimpleMatrix cameraMatrix = new SimpleMatrix(3, 3);
        JSONArray cameraMatrixArray = jsonData.getJSONArray("camera_matrix");
        for (int i = 0; i < cameraMatrix.numRows(); i++) {
            for (int j = 0; j < cameraMatrix.numCols(); j++) {
                cameraMatrix.set(i, j, cameraMatrixArray.getDouble(i * cameraMatrix.numCols() + j));
            }
        }

        SimpleMatrix cameraDistortion = new SimpleMatrix(8, 1);
        JSONArray cameraDistortionArray = jsonData.getJSONArray("distortion_coefficients");
        for (int i = 0; i < cameraDistortion.numRows(); i++) {
            cameraDistortion.set(i, cameraDistortionArray.getDouble(i));
        }

        return new Tuple<>(cameraMatrix, cameraDistortion);
    }

    public static Map<Integer, JSONObject> loadIdealMap(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            jsonBuilder.append(line);
        }
        reader.close();
        JSONObject jsonData = new JSONObject(jsonBuilder.toString());

        Map<Integer, JSONObject> idealMap = new HashMap<>();
        JSONArray tagsArray = jsonData.getJSONArray("tags");
        for (int i = 0; i < tagsArray.length(); i++) {
            JSONObject element = tagsArray.getJSONObject(i);
            int id = element.getInt("ID");
            idealMap.put(id, element);
        }

        return idealMap;
    }

    public static SimpleMatrix getTagTransform(Map<Integer, JSONObject> idealMap, int tagId) {
        SimpleMatrix transform = SimpleMatrix.identity(4);

        JSONObject tag = idealMap.get(tagId);
        JSONObject rotation = tag.getJSONObject("pose").getJSONObject("rotation").getJSONObject("quaternion");
        JSONObject translation = tag.getJSONObject("pose").getJSONObject("translation");

        double w = rotation.getDouble("W");
        double x = rotation.getDouble("X");
        double y = rotation.getDouble("Y");
        double z = rotation.getDouble("Z");

        SimpleMatrix rotationMatrix = new SimpleMatrix(3, 3);
        rotationMatrix.set(0, 0, 1 - 2 * y * y - 2 * z * z);
        rotationMatrix.set(0, 1, 2 * x * y - 2 * z * w);
        rotationMatrix.set(0, 2, 2 * x * z + 2 * y * w);
        rotationMatrix.set(1, 0, 2 * x * y + 2 * z * w);
        rotationMatrix.set(1, 1, 1 - 2 * x * x - 2 * z * z);
        rotationMatrix.set(1, 2, 2 * y * z - 2 * x * w);
        rotationMatrix.set(2, 0, 2 * x * z - 2 * y * w);
        rotationMatrix.set(2, 1, 2 * y * z + 2 * x * w);
        rotationMatrix.set(2, 2, 1 - 2 * x * x - 2 * y * y);

        transform.insertIntoThis(0, 0, rotationMatrix);
        transform.set(0, 3, translation.getDouble("x"));
        transform.set(1, 3, translation.getDouble("y"));
        transform.set(2, 3, translation.getDouble("z"));

        return transform;
    }

    public static SimpleMatrix estimateTagPose(
        double[][] tagDetection,
        SimpleMatrix cameraMatrix,
        SimpleMatrix cameraDistortion,
        double tagSize
    ) {
        Mat cameraMatrixCv = new Mat(3, 3, CvType.CV_64F);
        Mat cameraDistortionCv = new Mat(8, 1, CvType.CV_64F);

        for (int i = 0; i < cameraMatrix.numRows(); i++) {
            for (int j = 0; j < cameraMatrix.numCols(); j++) {
                cameraMatrixCv.put(i, j, cameraMatrix.get(i, j));
            }
        }

        for (int i = 0; i < cameraDistortion.numRows(); i++) {
            cameraDistortionCv.put(i, 0, cameraDistortion.get(i, 0));
        }

        List<Point> points2d = Arrays.asList(
            new Point(tagDetection[0][0], tagDetection[0][1]),
            new Point(tagDetection[1][0], tagDetection[1][1]),
            new Point(tagDetection[2][0], tagDetection[2][1]),
            new Point(tagDetection[3][0], tagDetection[3][1])
        );

        List<Point3> points3dBoxBase = Arrays.asList(
            new Point3(-tagSize / 2.0,  tagSize / 2.0, 0.0),
            new Point3( tagSize / 2.0,  tagSize / 2.0, 0.0),
            new Point3( tagSize / 2.0, -tagSize / 2.0, 0.0),
            new Point3(-tagSize / 2.0, -tagSize / 2.0, 0.0)
        );

        Mat rVec = new Mat();
        Mat tVec = new Mat();

        MatOfPoint3f objectPoints = new MatOfPoint3f();
        objectPoints.fromList(points3dBoxBase);

        MatOfPoint2f imagePoints = new MatOfPoint2f();
        imagePoints.fromList(points2d);

        MatOfDouble distCoeffs = new MatOfDouble();
        cameraDistortionCv.convertTo(distCoeffs, CvType.CV_64F);

        Calib3d.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrixCv,
            distCoeffs,
            rVec,
            tVec
        );

        Mat rMat = new Mat();
        Calib3d.Rodrigues(rVec, rMat);

        SimpleMatrix cameraToTag = new SimpleMatrix(4, 4);
        cameraToTag.set(0, 0, rMat.get(0, 0)[0]);
        cameraToTag.set(0, 1, rMat.get(0, 1)[0]);
        cameraToTag.set(0, 2, rMat.get(0, 2)[0]);
        cameraToTag.set(0, 3, tVec.get(0, 0)[0]);
        cameraToTag.set(1, 0, rMat.get(1, 0)[0]);
        cameraToTag.set(1, 1, rMat.get(1, 1)[0]);
        cameraToTag.set(1, 2, rMat.get(1, 2)[0]);
        cameraToTag.set(1, 3, tVec.get(1, 0)[0]);
        cameraToTag.set(2, 0, rMat.get(2, 0)[0]);
        cameraToTag.set(2, 1, rMat.get(2, 1)[0]);
        cameraToTag.set(2, 2, rMat.get(2, 2)[0]);
        cameraToTag.set(2, 3, tVec.get(2, 0)[0]);
        cameraToTag.set(3, 0, 0.0);
        cameraToTag.set(3, 1, 0.0);
        cameraToTag.set(3, 2, 0.0);
        cameraToTag.set(3, 3, 1.0);

        return cameraToTag;
    }
}
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
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.fitting.leastsquares.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.Pair;

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

        // Distortion coefficients
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

    public static void optimizePoses(List<Pose> poses, List<Constraint> constraints) {
        MultivariateJacobianFunction objectiveFunction = new MultivariateJacobianFunction() {
            @Override
            public Pair<RealVector, RealMatrix> value(RealVector point) {
                int numPoses = poses.size();
                int numConstraints = constraints.size();
                double[] residuals = new double[numConstraints * 6];
                double[][] jacobian = new double[numConstraints * 6][numPoses * 7];

                for (int i = 0; i < numPoses; i++) {
                    double[] poseData = point.toArray();
                    poses.get(i).p.set(0, poseData[i * 7]);
                    poses.get(i).p.set(1, poseData[i * 7 + 1]);
                    poses.get(i).p.set(2, poseData[i * 7 + 2]);
                    poses.get(i).q.set(0, poseData[i * 7 + 3]);
                    poses.get(i).q.set(1, poseData[i * 7 + 4]);
                    poses.get(i).q.set(2, poseData[i * 7 + 5]);
                    poses.get(i).q.set(3, poseData[i * 7 + 6]);
                }

                for (int k = 0; k < numConstraints; k++) {
                    Constraint constraint = constraints.get(k);
                    Pose poseBegin = poses.get(constraint.id_begin);
                    Pose poseEnd = poses.get(constraint.id_end);

                    // Compute residuals but its just a simple example
                    SimpleMatrix residual = poseBegin.p.minus(poseEnd.p);
                    System.arraycopy(residual.getDDRM().getData(), 0, residuals, k * 6, 3);

                    // Compute jacobian but its just a sipmle example
                    for (int j = 0; j < 3; j++) {
                        jacobian[k * 6 + j][constraint.id_begin * 7 + j] = 1.0;
                        jacobian[k * 6 + j][constraint.id_end * 7 + j] = -1.0;
                    }
                }

                RealVector residualsVector = new ArrayRealVector(residuals);
                RealMatrix jacobianMatrix = new Array2DRowRealMatrix(jacobian);

                return new Pair<>(residualsVector, jacobianMatrix);
            }
        };

        double[] initialGuess = new double[poses.size() * 7];
        for (int i = 0; i < poses.size(); i++) {
            System.arraycopy(poses.get(i).p.getDDRM().getData(), 0, initialGuess, i * 7, 3);
            System.arraycopy(poses.get(i).q.getDDRM().getData(), 0, initialGuess, i * 7 + 3, 4);
        }

        LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();

        LeastSquaresProblem problem = new LeastSquaresBuilder()
            .start(initialGuess)
            .model(objectiveFunction)
            .target(new double[constraints.size() * 6]) // 6 residuals per constraint
            .lazyEvaluation(false)
            .maxEvaluations(1000)
            .maxIterations(1000)
            .build();

        LeastSquaresOptimizer.Optimum optimum = optimizer.optimize(problem);

        double[] optimizedValues = optimum.getPoint().toArray();
        for (int i = 0; i < poses.size(); i++) {
            poses.get(i).p.set(0, optimizedValues[i * 7]);
            poses.get(i).p.set(1, optimizedValues[i * 7 + 1]);
            poses.get(i).p.set(2, optimizedValues[i * 7 + 2]);
            poses.get(i).q.set(0, optimizedValues[i * 7 + 3]);
            poses.get(i).q.set(1, optimizedValues[i * 7 + 4]);
            poses.get(i).q.set(2, optimizedValues[i * 7 + 5]);
            poses.get(i).q.set(3, optimizedValues[i * 7 + 6]);
        }
    }

    public static class Pose {
        public SimpleMatrix p;
        public SimpleMatrix q;

        public Pose(SimpleMatrix p, SimpleMatrix q) {
            this.p = p;
            this.q = q;
        }
    }

    public static class Constraint {
        public int id_begin;
        public int id_end;

        public Constraint(int id_begin, int id_end) {
            this.id_begin = id_begin;
            this.id_end = id_end;
        }
    }
}
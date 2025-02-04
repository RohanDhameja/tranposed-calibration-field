package com.example;

import java.util.*;
import java.io.*;
import org.json.*;
import org.ejml.simple.*;
import org.opencv.core.*;
import org.opencv.calib3d.*;
import org.opencv.imgproc.*;
import org.opencv.videoio.*;
import edu.wpi.first.apriltag.*;
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.fitting.leastsquares.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.Pair;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final String FRC_2025_FIELD_MAP_JSON = "{ \"tags\": ["
        + "{ \"ID\": 1, \"pose\": { \"translation\": { \"x\": 1.0, \"y\": 2.0, \"z\": 3.0 }, \"rotation\": { \"quaternion\": { \"W\": 1.0, \"X\": 0.0, \"Y\": 0.0, \"Z\": 0.0 } } } },"
        + "{ \"ID\": 2, \"pose\": { \"translation\": { \"x\": 4.0, \"y\": 5.0, \"z\": 6.0 }, \"rotation\": { \"quaternion\": { \"W\": 1.0, \"X\": 0.0, \"Y\": 0.0, \"Z\": 0.0 } } } },"
        + "{ \"ID\": 3, \"pose\": { \"translation\": { \"x\": 7.0, \"y\": 8.0, \"z\": 9.0 }, \"rotation\": { \"quaternion\": { \"W\": 1.0, \"X\": 0.0, \"Y\": 0.0, \"Z\": 0.0 } } } },"
        + "{ \"ID\": 4, \"pose\": { \"translation\": { \"x\": 10.0, \"y\": 11.0, \"z\": 12.0 }, \"rotation\": { \"quaternion\": { \"W\": 1.0, \"X\": 0.0, \"Y\": 0.0, \"Z\": 0.0 } } } }"
        + "] }";

    private static Mat cameraMatrix;
    private static Mat distCoeffs;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java Main <calibration_video> <video_file>");
            return;
        }

        String calibrationVideoPath = args[0];
        String videoFilePath = args[1];

        try {
            calibrateCamera(calibrationVideoPath);
            Map<Integer, SimpleMatrix> idealTransforms = loadIdealMap();
            processVideo(videoFilePath, idealTransforms);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void calibrateCamera(String videoFilePath) {
        List<Mat> imagePoints = new ArrayList<>();
        List<Mat> objectPoints = new ArrayList<>();
        MatOfPoint3f obj = new MatOfPoint3f();
        int boardWidth = 9;
        int boardHeight = 6;
        Size boardSize = new Size(boardWidth, boardHeight);

        for (int i = 0; i < boardHeight; i++) {
            for (int j = 0; j < boardWidth; j++) {
                obj.push_back(new MatOfPoint3f(new Point3(j, i, 0.0f)));
            }
        }

        VideoCapture capture = new VideoCapture(videoFilePath);
        if (!capture.isOpened()) {
            System.out.println("Error opening calibration video file");
            return;
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            MatOfPoint2f imageCorners = new MatOfPoint2f();
            boolean found = Calib3d.findChessboardCorners(gray, boardSize, imageCorners);

            if (found) {
                imagePoints.add(imageCorners);
                objectPoints.add(obj);
            }
        }

        capture.release();

        cameraMatrix = Mat.eye(3, 3, CvType.CV_64F);
        distCoeffs = Mat.zeros(5, 1, CvType.CV_64F);
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();

        Calib3d.calibrateCamera(objectPoints, imagePoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
    }

    public static Map<Integer, SimpleMatrix> loadIdealMap() throws IOException {
        JSONObject jsonData = new JSONObject(FRC_2025_FIELD_MAP_JSON);

        Map<Integer, SimpleMatrix> idealTransforms = new HashMap<>();
        JSONArray tagsArray = jsonData.getJSONArray("tags");
        for (int i = 0; i < tagsArray.length(); i++) {
            JSONObject tag = tagsArray.getJSONObject(i);
            SimpleMatrix H_world_tag = SimpleMatrix.identity(4);

            H_world_tag.set(0, 3, tag.getJSONObject("pose").getJSONObject("translation").getDouble("x"));
            H_world_tag.set(1, 3, tag.getJSONObject("pose").getJSONObject("translation").getDouble("y"));
            H_world_tag.set(2, 3, tag.getJSONObject("pose").getJSONObject("translation").getDouble("z"));

            double[] quaternion = {
                tag.getJSONObject("pose").getJSONObject("rotation").getJSONObject("quaternion").getDouble("W"),
                tag.getJSONObject("pose").getJSONObject("rotation").getJSONObject("quaternion").getDouble("X"),
                tag.getJSONObject("pose").getJSONObject("rotation").getJSONObject("quaternion").getDouble("Y"),
                tag.getJSONObject("pose").getJSONObject("rotation").getJSONObject("quaternion").getDouble("Z")
            };

            SimpleMatrix rotationMatrix = quaternionToMatrix(quaternion);
            H_world_tag.insertIntoThis(0, 0, rotationMatrix);

            idealTransforms.put(tag.getInt("ID"), H_world_tag);
        }

        return idealTransforms;
    }

    private static SimpleMatrix quaternionToMatrix(double[] q) {
        double w = q[0];
        double x = q[1];
        double y = q[2];
        double z = q[3];

        double[][] data = {
            {1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w},
            {2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w},
            {2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y}
        };

        return new SimpleMatrix(data);
    }

    public static void processVideo(String videoFilePath, Map<Integer, SimpleMatrix> idealTransforms) {
        // Load the video
        VideoCapture capture = new VideoCapture(videoFilePath);
        if (!capture.isOpened()) {
            System.out.println("Error opening video file");
            return;
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            // Process each frame to detect AprilTags and extract their locations
            List<DetectedTag> detectedTags = detectAprilTags(frame);

            // Use the reference tag to find the locations of the rest
            if (!detectedTags.isEmpty()) {
                DetectedTag referenceTag = detectedTags.get(0); // Assume the first detected tag is the reference tag
                SimpleMatrix referenceTransform = referenceTag.transform;

                for (DetectedTag detectedTag : detectedTags) {
                    SimpleMatrix relativeTransform = referenceTransform.invert().mult(detectedTag.transform);
                    detectedTag.transform = relativeTransform;
                }

                // Compare detected tags with ideal map
                compareWithIdealMap(detectedTags, idealTransforms);
            }

            // Display the frame (optional)
            // HighGui.imshow("Frame", frame);
            // if (HighGui.waitKey(30) >= 0) break;
        }

        capture.release();
        // HighGui.destroyAllWindows();
    }

    public static List<DetectedTag> detectAprilTags(Mat frame) {
        List<DetectedTag> detectedTags = new ArrayList<>();

        // Convert the frame to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        // Initialize the AprilTag detector
        AprilTagDetector detector = new AprilTagDetector();
        detector.addFamily("tag36h11");

        // Detect tags
        List<AprilTagDetection> detections = detector.detect(gray);

        // Process detections
        for (AprilTagDetection detection : detections) {
            int id = detection.getId();
            double[] translation = detection.getCenter();
            double[] rotation = detection.getRotation();

            // Create transformation matrix
            SimpleMatrix transform = SimpleMatrix.identity(4);
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    transform.set(row, col, rotation[row * 3 + col]);
                }
                transform.set(row, 3, translation[row]);
            }

            detectedTags.add(new DetectedTag(id, transform));
        }

        return detectedTags;
    }

    public static void compareWithIdealMap(List<DetectedTag> detectedTags, Map<Integer, SimpleMatrix> idealTransforms) {
        // Implement comparison logic
        for (DetectedTag detectedTag : detectedTags) {
            SimpleMatrix idealTransform = idealTransforms.get(detectedTag.id);
            if (idealTransform != null) {
                // Compare detectedTag.transform with idealTransform
                // Add code to visualize or log the comparison results
                System.out.println("Tag ID: " + detectedTag.id);
                System.out.println("Detected Transform: " + detectedTag.transform);
                System.out.println("Ideal Transform: " + idealTransform);
                System.out.println("Difference: " + idealTransform.minus(detectedTag.transform));
            }
        }
    }

    public static class DetectedTag {
        public int id;
        public SimpleMatrix transform;

        public DetectedTag(int id, SimpleMatrix transform) {
            this.id = id;
            this.transform = transform;
        }
    }
}
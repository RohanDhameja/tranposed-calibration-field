package com.example;

import java.io.*;
import java.util.*;
import org.json.*;
import org.ejml.simple.*;
import javafx.application.Application;
import javafx.scene.*;
import javafx.scene.paint.*;
import javafx.scene.shape.*;
import javafx.stage.Stage;

public class VisualizerFX extends Application {
    private static final double[][] CUBE_VERTICES = {
        {0, 0, 0},
        {0.1651, 0, 0},
        {0.1651, 0.1651, 0},
        {0, 0.1651, 0},
        {0, 0, 0.1651},
        {0.1651, 0, 0.1651},
        {0.1651, 0.1651, 0.1651},
        {0, 0.1651, 0.1651}
    };

    private static final int[][] CUBE_EDGES = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    private static Map<Integer, SimpleMatrix> idealTransforms;
    private static Map<Integer, SimpleMatrix> observedTransforms;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java VisualizerFX <ideal_map.json> <observed_map.json>");
            return;
        }

        try {
            idealTransforms = loadMap(args[0]);
            observedTransforms = loadMap(args[1]);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        Group root = new Group();
        Scene scene = new Scene(root, 800, 600, true);
        scene.setFill(Color.BLACK);

        PerspectiveCamera camera = new PerspectiveCamera(true);
        camera.setTranslateZ(-5);
        scene.setCamera(camera);

        for (Map.Entry<Integer, SimpleMatrix> entry : observedTransforms.entrySet()) {
            plotTransformation(entry.getValue(), root, Color.LIME);
        }

        for (Map.Entry<Integer, SimpleMatrix> entry : idealTransforms.entrySet()) {
            plotTransformation(entry.getValue(), root, Color.RED);
        }

        primaryStage.setTitle("3D Transformations");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private static Map<Integer, SimpleMatrix> loadMap(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            jsonBuilder.append(line);
        }
        reader.close();
        JSONObject jsonData = new JSONObject(jsonBuilder.toString());

        Map<Integer, SimpleMatrix> transforms = new HashMap<>();
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

            transforms.put(tag.getInt("ID"), H_world_tag);
        }

        return transforms;
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

    private static void plotTransformation(SimpleMatrix H, Group root, Color color) {
        SimpleMatrix transformedVertices = H.mult(new SimpleMatrix(new double[][]{
            {CUBE_VERTICES[0][0], CUBE_VERTICES[1][0], CUBE_VERTICES[2][0], CUBE_VERTICES[3][0], CUBE_VERTICES[4][0], CUBE_VERTICES[5][0], CUBE_VERTICES[6][0], CUBE_VERTICES[7][0]},
            {CUBE_VERTICES[0][1], CUBE_VERTICES[1][1], CUBE_VERTICES[2][1], CUBE_VERTICES[3][1], CUBE_VERTICES[4][1], CUBE_VERTICES[5][1], CUBE_VERTICES[6][1], CUBE_VERTICES[7][1]},
            {CUBE_VERTICES[0][2], CUBE_VERTICES[1][2], CUBE_VERTICES[2][2], CUBE_VERTICES[3][2], CUBE_VERTICES[4][2], CUBE_VERTICES[5][2], CUBE_VERTICES[6][2], CUBE_VERTICES[7][2]},
            {1, 1, 1, 1, 1, 1, 1, 1}
        }));

        for (int[] edge : CUBE_EDGES) {
            Line line = new Line(
                transformedVertices.get(0, edge[0]), transformedVertices.get(1, edge[0]),
                transformedVertices.get(0, edge[1]), transformedVertices.get(1, edge[1])
            );
            line.setStroke(color);
            root.getChildren().add(line);
        }

        Sphere sphere = new Sphere(0.01);
        sphere.setTranslateX(H.get(0, 3));
        sphere.setTranslateY(H.get(1, 3));
        sphere.setTranslateZ(H.get(2, 3));
        sphere.setMaterial(new PhongMaterial(color));
        root.getChildren().add(sphere);
    }
}
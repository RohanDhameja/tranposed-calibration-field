package com.example;

import org.ejml.simple.SimpleMatrix;

public class Pose {
    public SimpleMatrix p;
    public SimpleMatrix q;

    public Pose() {
        this.p = new SimpleMatrix(3, 1);
        this.q = new SimpleMatrix(4, 1);
    }
}
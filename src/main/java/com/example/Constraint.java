package com.example;

public class Constraint {
    public int id_begin;
    public int id_end;
    public Pose t_begin_end;

    public Constraint() {
        this.t_begin_end = new Pose();
    }
}
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ThreeNN")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("dist: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("ThreeInterpolate")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // (b,m,c)
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // (b,n,3)
        c->WithRank(c->input(1), 3, &dims2);
        // (b,n,c)
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("ThreeInterpolateGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

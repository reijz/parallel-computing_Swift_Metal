//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


kernel void initialize(const device float *batch[[buffer(1)]],
                       const device float *parameters[[buffer(2)]],
                       device float *initValue [[buffer(0)]],
                       uint id [[ thread_position_in_grid ]]) {

    uint idCurrent = batch[0]*batch[1]+id;
    uint idParent = idCurrent - batch[0];
    initValue[idCurrent] = initValue[idParent] + 1;//parameters[2]; // salvage value per unit

}


kernel void iterate(const device float *batch[[buffer(2)]],
                    const device float *parameters[[buffer(3)]],
                    const device float *inVector [[buffer(0)]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {

    uint idCurrent = batch[0]*batch[1]+id;
    uint idParent = idCurrent - batch[0];
    outVector[idCurrent] = inVector[idCurrent];
}


// to test the understanding of thread related concepts
kernel void testThread(device float *result [[ buffer(0) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint i [[ thread_position_in_threadgroup ]],
                    uint w [[ threadgroup_position_in_grid ]],
                    uint S [[ threads_per_threadgroup ]]) {

    if (id == w*S+i)
        result[id] = id;
    else
        result[id] = 0;
}

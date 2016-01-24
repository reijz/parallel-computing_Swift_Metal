//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

__constant int unitSize = 4;

kernel void initialize(const device float *transmit[[buffer(0)]],
                       device float *initValue [[buffer(1)]],
                       uint id [[ thread_position_in_grid ]]) {

//    batchSize*k*unitSize
    int shift = transmit[1]*transmit[2];
    
    initValue[shift+id] = shift+id;

}


kernel void iterate(//const device float *parameters[[buffer(0)]],
                    const device float *inVector [[buffer(1)]],
                    device float *outVector [[ buffer(2) ]],
                    uint id [[ thread_position_in_grid ]]) {
    
    outVector[id] = inVector[id];
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

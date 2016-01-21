//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void initialize(const device float *parameter[[buffer(2)]],
                       device float *initValue [[buffer(0)]],
                       uint id [[ thread_position_in_grid ]],
                       uint i [[thread_position_in_threadgroup]],
                       uint w [[threadgroup_position_in_grid]],
                       uint S [[threads_per_threadgroup]]) {
    initValue[id] = parameter[2]*id;

}


kernel void iterate(const device float *parameter[[buffer(2)]],
                    const device float *inVector [[buffer(0)]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint i [[thread_position_in_threadgroup]],
                    uint w [[threadgroup_position_in_grid]],
                    uint S [[threads_per_threadgroup]]) {
    
        outVector[id] = parameter[0]*inVector[id];
}

kernel void testThread(device float *result [[ buffer(0) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint i [[thread_position_in_threadgroup]],
                    uint w [[threadgroup_position_in_grid]],
                    uint S [[threads_per_threadgroup]]) {

    if (id == w*S+i)
        result[id] = id;
    else
        result[id] = 0;
}

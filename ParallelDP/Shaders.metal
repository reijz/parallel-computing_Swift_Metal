//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

// Parameters on the device needs
__constant float holdingCost = 1;
__constant float orderCost = 1.5;
__constant float discountRate = 2;
__constant float price = 1.5;
__constant float desposalCost = 1.5;
__constant float salvageValue = 1.5;


kernel void initialize(device float *initValue [[buffer(0)]],
                       uint id [[ thread_position_in_grid ]],
                       uint i [[thread_position_in_threadgroup]],
                       uint w [[threadgroup_position_in_grid]],
                       uint S [[threads_per_threadgroup]]) {
    initValue[id] = orderCost*id;

}


kernel void iterate(const device float *inVector [[buffer(0)]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint i [[ thread_position_in_threadgroup ]],
                    uint w [[ threadgroup_position_in_grid ]],
                    uint S [[ threads_per_threadgroup ]]) {
    
        outVector[id] = discountRate*inVector[id];
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

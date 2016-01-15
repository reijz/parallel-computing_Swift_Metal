//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(device float *outVector [[ buffer(0) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint i [[thread_position_in_threadgroup]],
                    uint w [[threadgroup_position_in_grid]],
                    uint S [[threads_per_threadgroup]]) {

    if (id == w*S+i)
        outVector[id] = id;
    else
        outVector[id] = 0;
}

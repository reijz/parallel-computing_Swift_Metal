//
//  Shaders.metal
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#define max_dimension 13

kernel void initialize(const device uint *batch[[buffer(1)]],
                       const device float *parameters[[buffer(2)]],
                       device float *initValue [[buffer(0)]],
                       uint id [[ thread_position_in_grid ]]) {

    // get the parameters
    float salvageValue = parameters[2];

    // find current and parend id
    uint idCurrent = batch[0]*batch[1]+id;
    uint idParent = idCurrent - batch[0];
    
    initValue[idCurrent] = initValue[idParent] + salvageValue; // discountRate * orderCost - holdingCost;
}

// compute the value funciton of the optimal policy
kernel void iterate(const device uint *batch[[buffer(4)]],
                    const device float *parameters[[buffer(5)]],
                    const device float *distribution[[buffer(6)]],
                    const device float *inVector [[buffer(0)]],
                    device float *outVector [[ buffer(1) ]],
                    device float *deplete[[buffer(2)]],
                    device float *order[[buffer(3)]],
                    uint id [[ thread_position_in_grid ]]) {
    
    // get the parameters
    int K = int(parameters[0]), L = int(parameters[1]);
    int max_demand = int(parameters[8]);
    float salvageValue = parameters[2];
    float holdingCost = parameters[3];
    float orderCost = parameters[4];
    float disposalCost = parameters[5];
    float discountRate = parameters[6];
    float price = parameters[7];
    
    // find current and parend id
    uint idCurrent = batch[0]*batch[1]+id;
    uint idParent = idCurrent - batch[0];
    
    // prepare a vector for decode
    int idState[max_dimension + 1];
    
    // range of optimization
    int min_deplete = 0, max_deplete = 1;
    int min_order = 0, max_order = K;

    if (idCurrent != 0){
        min_deplete = int(deplete[idParent]) + int(deplete[idParent] != 0.);
        max_deplete = int(deplete[idParent]) + 2;
        int min_order_1 = int(order[idParent]) + int(deplete[idParent] != 0.) - 1;
        min_order = min_order_1 * int(min_order_1 >= 0);
        max_order = int(order[idParent]) + 1;
    }
    
    int opt_deplete = 0;
    int opt_order = 0;
    float opt_value = 0.;
    float state_value = 0.;
    
    
    for (int i = min_deplete; i < max_deplete; i++){
        for (int j = min_order; j < max_order; j++){
            state_value= 0.;
            for (int d = 0; d < max_demand; d++){
                // decode idCurrent into idState
                int idSum= 0, index= idCurrent;
                for (int l = L - 1; l >= 0; l--) {
                   idState[l] = index % K;
                   idSum += idState[l];
                   index /= K;
                }
                idState[L] = 0;
                //deplete i units from idState
                int remain_deplete = i;
                for (int l = 0; l < L; l++) {
                    if (remain_deplete <= idState[l]) {
                        idState[l] -= remain_deplete;
                        break;
                    } else {
                        remain_deplete -= idState[l];
                        idState[l] = 0;
                    }
                }
                //holding cost incurred
                int hold = idSum - i;
                //order j units
                idState[L]= j;
                //sell d units from idState
                int sell = 0, remain_sell = d;
                for (int l = 0; l < L+ 1; l++) {
                    if (remain_sell <= idState[l]) {
                        sell += remain_sell;
                        idState[l] -= remain_sell;
                        break;
                    } else {
                        remain_sell -= idState[l];
                        sell += idState[l];
                        idState[l] = 0;
                    }
                }
                //dispose expired terms
                int dispose = idState[0];
                idState[0]= 0;
                //get the index of the future state
                int future = 0;
                for (int l = 1; l < L + 1; l++) {
                    future *= K;
                    future += idState[l];
                }
                //get the value with respect to i, j, d
                float state_value_sample = salvageValue * i
                                           - holdingCost * hold
                                           + discountRate * (-orderCost * j
                                                             + price * sell
                                                             - disposalCost * dispose
                                                             + inVector[future]);
                state_value += (state_value_sample * distribution[d]);
            }
            if (state_value > opt_value + 1e-6){
                opt_value = state_value;
                opt_deplete = i;
                opt_order = j;
            }
        }
    }
    
    outVector[idCurrent] = opt_value; 
    deplete[idCurrent] = float(opt_deplete);
    order[idCurrent] = float(opt_order);
    
}

// compute the value function of the fluid policy
kernel void iterate_fluid(const device uint *batch[[buffer(4)]],
                    const device float *parameters[[buffer(5)]],
                    const device float *distribution[[buffer(6)]],
                    const device float *inVector [[buffer(0)]],
                    device float *outVector [[ buffer(1) ]],
                    device float *deplete[[buffer(2)]],
                    device float *order[[buffer(3)]],
                    uint id [[ thread_position_in_grid ]]) {
    
    // get the parameters
    int K = int(parameters[0]), L = int(parameters[1]), numPeriods= int(parameters[10]);
    int max_demand = int(parameters[8]);
    int mean_demand = int(parameters[9]);
    float salvageValue = parameters[2];
    float holdingCost = parameters[3];
    float orderCost = parameters[4];
    float disposalCost = parameters[5];
    float discountRate = parameters[6];
    float price = parameters[7];
    
    int t = batch[2];
    
    // find current and parend id
    uint idCurrent = batch[0]*batch[1]+id;
    
    // prepare a vector for decode
    int idState[max_dimension + 1];
    
    int idSum= 0, index= idCurrent;
    for (int l = L - 1; l >= 0; l--) {
        idState[l] = index % K;
        idSum += idState[l];
        index /= K;
    }
    idState[L] = 0;
    
    int fluid_deplete = 0, fluid_order = 0;
    // determine the action for a specific state
    if (t < numPeriods -1) {
        if (mean_demand - idSum > 1e-6){
            fluid_order = int(mean_demand - idSum);
        }
    }
    else {
        int idPartialSum = 0;
        for (int l = 1; l < L + 1; l++) {
            idPartialSum += idState[l-1];
            if (idPartialSum - l* mean_demand > fluid_deplete + 1e-6){
                fluid_deplete = idPartialSum - l* mean_demand;
            }
        }
        if (mean_demand + fluid_deplete - idSum > 1e-6){
            fluid_order = int(mean_demand + fluid_deplete - idSum);
        }
        
    }
    float opt_value = 0, state_value = 0;
    
    for (int i = fluid_deplete; i < fluid_deplete + 1; i++){
        for (int j = fluid_order; j < fluid_order + 1; j++){
            state_value= 0.;
            for (int d = 0; d < max_demand; d++){
                // decode idCurrent into idState
                int idSum= 0, index= idCurrent;
                for (int l = L - 1; l >= 0; l--) {
                    idState[l] = index % K;
                    idSum += idState[l];
                    index /= K;
                }
                idState[L] = 0;
                //deplete i units from idState
                int remain_deplete = i;
                for (int l = 0; l < L; l++) {
                    if (remain_deplete <= idState[l]) {
                        idState[l] -= remain_deplete;
                        break;
                    } else {
                        remain_deplete -= idState[l];
                        idState[l] = 0;
                    }
                }
                //holding cost incurred
                int hold = idSum - i;
                //order j units
                idState[L]= j;
                //sell d units from idState
                int sell = 0, remain_sell = d;
                for (int l = 0; l < L+ 1; l++) {
                    if (remain_sell <= idState[l]) {
                        sell += remain_sell;
                        idState[l] -= remain_sell;
                        break;
                    } else {
                        remain_sell -= idState[l];
                        sell += idState[l];
                        idState[l] = 0;
                    }
                }
                //dispose expired terms
                int dispose = idState[0];
                idState[0]= 0;
                //get the index of the future state
                int future = 0;
                for (int l = 1; l < L + 1; l++) {
                    future *= K;
                    future += idState[l];
                }
                //get the value with respect to i, j, d
                float state_value_sample = salvageValue * i
                - holdingCost * hold
                + discountRate * (-orderCost * j
                                  + price * sell
                                  - disposalCost * dispose
                                  + inVector[future]);
                state_value += (state_value_sample * distribution[d]);
            }
            if (state_value > opt_value + 1e-6){
                opt_value = state_value;
            }
        }
    }
    
    outVector[idCurrent] = opt_value;
    deplete[idCurrent] = float(fluid_deplete);
    order[idCurrent] = float(fluid_order);
    
}

// compute the value function of the NVP policy
kernel void iterate_NVP(const device uint *batch[[buffer(4)]],
                          const device float *parameters[[buffer(5)]],
                          const device float *distribution[[buffer(6)]],
                          const device float *inVector [[buffer(0)]],
                          device float *outVector [[ buffer(1) ]],
                          device float *deplete[[buffer(2)]],
                          device float *order[[buffer(3)]],
                          uint id [[ thread_position_in_grid ]]) {
    
    // get the parameters
    int K = int(parameters[0]), L = int(parameters[1]), numPeriods= int(parameters[10]);
    int max_demand = int(parameters[8]);
    float salvageValue = parameters[2];
    float holdingCost = parameters[3];
    float orderCost = parameters[4];
    float disposalCost = parameters[5];
    float discountRate = parameters[6];
    float price = parameters[7];
    
    int t = batch[2];
    
    // find current and parend id
    uint idCurrent = batch[0]*batch[1]+id;
    uint idParent = idCurrent - batch[0];
    
    // prepare a vector for decode
    int idState[max_dimension + 1];
    
    int idSum= 0, index= idCurrent;
    for (int l = L - 1; l >= 0; l--) {
        idState[l] = index % K;
        idSum += idState[l];
        index /= K;
    }
    idState[L] = 0;
    
    int min_deplete = 0, max_deplete = 1;
    int min_order = 0, max_order = K;
    
    if (t < numPeriods- 1) {
        if (K-1 - idSum > 1e-6){
            min_order = int(K-1 - idSum);
            max_order = int(K-1 - idSum) + 1;
        }
    }
    else {
        if (idCurrent != 0){
            min_deplete = int(deplete[idParent]) + int(deplete[idParent] != 0.);
            max_deplete = int(deplete[idParent]) + 2;
            int min_order_1 = int(order[idParent]) + int(deplete[idParent] != 0.) - 1;
            min_order = min_order_1 * int(min_order_1 >= 0);
            max_order = int(order[idParent]) + 1;
        }
    }

    int NVP_deplete = 0;
    int NVP_order = 0;
    float opt_value = 0.;
    float state_value = 0.;
        
    for (int i = min_deplete; i < max_deplete; i++){
        for (int j = min_order; j < max_order; j++){
            state_value= 0.;
            for (int d = 0; d < max_demand; d++){
                // decode idCurrent into idState
                int idSum= 0, index= idCurrent;
                for (int l = L - 1; l >= 0; l--) {
                    idState[l] = index % K;
                    idSum += idState[l];
                    index /= K;
                }
                idState[L] = 0;
                //deplete i units from idState
                int remain_deplete = i;
                for (int l = 0; l < L; l++) {
                    if (remain_deplete <= idState[l]) {
                        idState[l] -= remain_deplete;
                        break;
                    } else {
                        remain_deplete -= idState[l];
                        idState[l] = 0;
                    }
                }
                //holding cost incurred
                int hold = idSum - i;
                //order j units
                idState[L]= j;
                //sell d units from idState
                int sell = 0, remain_sell = d;
                for (int l = 0; l < L+ 1; l++) {
                    if (remain_sell <= idState[l]) {
                        sell += remain_sell;
                        idState[l] -= remain_sell;
                        break;
                    } else {
                        remain_sell -= idState[l];
                        sell += idState[l];
                        idState[l] = 0;
                    }
                }
                //dispose expired terms
                int dispose = idState[0];
                idState[0]= 0;
                //get the index of the future state
                int future = 0;
                for (int l = 1; l < L + 1; l++) {
                    future *= K;
                    future += idState[l];
                }
                //get the value with respect to i, j, d
                float state_value_sample = salvageValue * i
                - holdingCost * hold
                + discountRate * (-orderCost * j
                                  + price * sell
                                  - disposalCost * dispose
                                  + inVector[future]);
                state_value += (state_value_sample * distribution[d]);
            }
            if (state_value > opt_value + 1e-6){
                opt_value = state_value;
                NVP_deplete = i;
                NVP_order = j;
            }
        }
    }
    
    outVector[idCurrent] = opt_value;
    deplete[idCurrent] = float(NVP_deplete);
    order[idCurrent] = float(NVP_order);
    
}

// compute the value function of the NVP_variant policy
kernel void iterate_NVP_1(const device uint *batch[[buffer(4)]],
                          const device float *parameters[[buffer(5)]],
                          const device float *distribution[[buffer(6)]],
                          const device float *inVector [[buffer(0)]],
                          device float *outVector [[ buffer(1) ]],
                          device float *deplete[[buffer(2)]],
                          device float *order[[buffer(3)]],
                          uint id [[ thread_position_in_grid ]]) {
    
    // get the parameters
    int K = int(parameters[0]), L = int(parameters[1]);
    int max_demand = int(parameters[8]);
    float salvageValue = parameters[2];
    float holdingCost = parameters[3];
    float orderCost = parameters[4];
    float disposalCost = parameters[5];
    float discountRate = parameters[6];
    float price = parameters[7];
    
    int deplete_down_to = int(parameters[11]);
    
    // find current and parend id
    uint idCurrent = batch[0]*batch[1]+id;
    
    // prepare a vector for decode
    int idState[max_dimension + 1];
    
    int idSum= 0, index= idCurrent;
    for (int l = L - 1; l >= 0; l--) {
        idState[l] = index % K;
        idSum += idState[l];
        index /= K;
    }
    idState[L] = 0;
    
    int NVP_1_deplete = 0, NVP_1_order = 0;
    // determine the action for a specific state
    if (idState[0] > deplete_down_to) {
        NVP_1_deplete = idState[0]- deplete_down_to;
    }
    if (K-1 + NVP_1_deplete - idSum > 1e-6){
            NVP_1_order = int(K- 1 + NVP_1_deplete - idSum);
    }
        
    
    float opt_value = 0, state_value = 0;
    
    for (int i = NVP_1_deplete; i < NVP_1_deplete + 1; i++){
        for (int j = NVP_1_order; j < NVP_1_order + 1; j++){
            state_value= 0.;
            for (int d = 0; d < max_demand; d++){
                // decode idCurrent into idState
                int idSum= 0, index= idCurrent;
                for (int l = L - 1; l >= 0; l--) {
                    idState[l] = index % K;
                    idSum += idState[l];
                    index /= K;
                }
                idState[L] = 0;
                //deplete i units from idState
                int remain_deplete = i;
                for (int l = 0; l < L; l++) {
                    if (remain_deplete <= idState[l]) {
                        idState[l] -= remain_deplete;
                        break;
                    } else {
                        remain_deplete -= idState[l];
                        idState[l] = 0;
                    }
                }
                //holding cost incurred
                int hold = idSum - i;
                //order j units
                idState[L]= j;
                //sell d units from idState
                int sell = 0, remain_sell = d;
                for (int l = 0; l < L+ 1; l++) {
                    if (remain_sell <= idState[l]) {
                        sell += remain_sell;
                        idState[l] -= remain_sell;
                        break;
                    } else {
                        remain_sell -= idState[l];
                        sell += idState[l];
                        idState[l] = 0;
                    }
                }
                //dispose expired terms
                int dispose = idState[0];
                idState[0]= 0;
                //get the index of the future state
                int future = 0;
                for (int l = 1; l < L + 1; l++) {
                    future *= K;
                    future += idState[l];
                }
                //get the value with respect to i, j, d
                float state_value_sample = salvageValue * i
                - holdingCost * hold
                + discountRate * (-orderCost * j
                                  + price * sell
                                  - disposalCost * dispose
                                  + inVector[future]);
                state_value += (state_value_sample * distribution[d]);
            }
            if (state_value > opt_value + 1e-6){
                opt_value = state_value;
            }
        }
    }
    
    outVector[idCurrent] = opt_value;
    deplete[idCurrent] = float(NVP_1_deplete);
    order[idCurrent] = float(NVP_1_order);
    
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

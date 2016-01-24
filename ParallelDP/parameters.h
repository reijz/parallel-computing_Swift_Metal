//
//  parameters.h
//  ParallelDP
//
//  Created by Jiheng Zhang on 24/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

#ifndef parameters_h
#define parameters_h

// Parameters on the device needs
__constant int K = 8;  // capacity
__constant int L = 8;  // dimension
__constant float holdingCost = 1;
__constant float orderCost = 1.5;
__constant float discountRate = 2;
__constant float price = 1.5;
__constant float desposalCost = 1.5;
__constant float salvageValue = 1.5;

#endif /* parameters_h */

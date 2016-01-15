//
//  main.swift
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

import Foundation
import MetalKit

// parameter setting for DP

let K = 8
let L = 3

let numberOfStates = Int(pow(Float(K), Float(L)))
let unitSize = sizeof(Float)
let resultBufferSize = numberOfStates*unitSize

// hardcoded to 512 for now (recommendation: read about threadExecutionWidth)
let threadExecutionWidth = 512
let numThreadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
let numGroups = MTLSize(width:(resultBufferSize+threadExecutionWidth-1)/threadExecutionWidth, height:1, depth:1)

// Initialize Metal

// var devices = MTLCopyAllDevices()
// let deviceNumber = devices.count

var device: MTLDevice! = MTLCreateSystemDefaultDevice()

var commandQueue: MTLCommandQueue! = device.newCommandQueue()

var defaultLibrary: MTLLibrary! = device.newDefaultLibrary()

var commandBuffer: MTLCommandBuffer! = commandQueue.commandBuffer()

var computeCommandEncoder = commandBuffer.computeCommandEncoder()



// set up a compute pipeline with iteration and add it to encoder
let sigmoidProgram = defaultLibrary.newFunctionWithName("sigmoid")
var computePipelineFilter = try device.newComputePipelineStateWithFunction(sigmoidProgram!)
computeCommandEncoder.setComputePipelineState(computePipelineFilter)


// Prepare shared memory
let resourceOption = MTLResourceOptions()
var outVectorBuffer = device.newBufferWithLength(resultBufferSize, options: resourceOption)
computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, atIndex: 0)


computeCommandEncoder.dispatchThreadgroups(numThreadsPerGroup, threadsPerThreadgroup: numGroups)
computeCommandEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()



// a. Get GPU data
// outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
var data = NSData(bytesNoCopy: outVectorBuffer.contents(), length: resultBufferSize, freeWhenDone: false)
// b. prepare Swift array large enough to receive data from GPU
var finalResultArray = [Float](count: numberOfStates, repeatedValue: 0)

// c. get data from GPU into Swift array
data.getBytes(&finalResultArray, length:resultBufferSize)

print(finalResultArray)
// print(statesResultA)


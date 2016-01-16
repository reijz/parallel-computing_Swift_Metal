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
var device: MTLDevice! = MTLCreateSystemDefaultDevice()
// Build command queue
var commandQueue: MTLCommandQueue! = device.newCommandQueue()

// Allocate memory on device
let resourceOption = MTLResourceOptions()
var bufferA = device.newBufferWithLength(resultBufferSize, options: resourceOption)
var bufferB = device.newBufferWithLength(resultBufferSize, options: resourceOption)

// Get functions from Shaders and add to MTL library
var defaultLibrary: MTLLibrary! = device.newDefaultLibrary()
let initDP = defaultLibrary.newFunctionWithName("initialize")
let iterateDP = defaultLibrary.newFunctionWithName("iterate")

var commandBufferInitDP: MTLCommandBuffer! = commandQueue.commandBuffer()
var encoderInitDP = commandBufferInitDP.computeCommandEncoder()
var pipelineFilterInit = try device.newComputePipelineStateWithFunction(initDP!)
encoderInitDP.setComputePipelineState(pipelineFilterInit)
encoderInitDP.setBuffer(bufferA, offset: 0, atIndex: 0)
encoderInitDP.dispatchThreadgroups(numThreadsPerGroup, threadsPerThreadgroup: numGroups)
encoderInitDP.endEncoding()
commandBufferInitDP.commit()
commandBufferInitDP.waitUntilCompleted()


var commandBufferIterateDP: MTLCommandBuffer! = commandQueue.commandBuffer()
var encoderIterateDP = commandBufferIterateDP.computeCommandEncoder()
var pipelineFilterIterate = try device.newComputePipelineStateWithFunction(iterateDP!)
encoderIterateDP.setComputePipelineState(pipelineFilterIterate)
encoderIterateDP.setBuffer(bufferA, offset: 0, atIndex: 0)
encoderIterateDP.setBuffer(bufferB, offset: 0, atIndex: 1)
encoderIterateDP.dispatchThreadgroups(numThreadsPerGroup, threadsPerThreadgroup: numGroups)
encoderIterateDP.endEncoding()
commandBufferIterateDP.commit()
commandBufferIterateDP.waitUntilCompleted()


// a. Get GPU data
var data = NSData(bytesNoCopy: bufferB.contents(), length: resultBufferSize, freeWhenDone: false)
// b. prepare Swift array large enough to receive data from GPU
var finalResultArray = [Float](count: numberOfStates, repeatedValue: 0)
// c. get data from GPU into Swift array
data.getBytes(&finalResultArray, length:resultBufferSize)

print(finalResultArray)


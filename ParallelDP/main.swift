//
//  main.swift
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

import Foundation
import MetalKit

// Initialize Metal

// var devices = MTLCopyAllDevices()
// let deviceNumber = devices.count

var device: MTLDevice! = MTLCreateSystemDefaultDevice()

var commandQueue: MTLCommandQueue! = device.newCommandQueue()

var defaultLibrary: MTLLibrary! = device.newDefaultLibrary()

var commandBuffer: MTLCommandBuffer! = commandQueue.commandBuffer()

var computeCommandEncoder = commandBuffer.computeCommandEncoder()


// set up a compute pipeline with Sigmoid function and add it to encoder
let sigmoidProgram = defaultLibrary.newFunctionWithName("sigmoid")
var computePipelineFilter = try device.newComputePipelineStateWithFunction(sigmoidProgram!)
computeCommandEncoder.setComputePipelineState(computePipelineFilter)


// Prepare input data
var myvector = [Float](count: 123456, repeatedValue: 0)
for (index, value) in myvector.enumerate() {
    myvector[index] = Float(index)
}


// calculate byte length of input data - myvector
var myvectorByteLength = myvector.count*sizeofValue(myvector[0])
var resourceOption = MTLResourceOptions()

// b. create a MTLBuffer - input data that the GPU and Metal and produce

var inVectorBuffer = device.newBufferWithBytes(&myvector, length: myvectorByteLength, options: resourceOption)

// c. set the input vector for the Sigmoid() function, e.g. inVector
//    atIndex: 0 here corresponds to buffer(0) in the Sigmoid function
computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, atIndex: 0)

// d. create the output vector for the Sigmoid() function, e.g. outVector
//    atIndex: 1 here corresponds to buffer(1) in the Sigmoid function
var resultdata = [Float](count:myvector.count, repeatedValue: 0)
var outVectorBuffer = device.newBufferWithBytes(&resultdata, length: myvectorByteLength, options: resourceOption)
computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, atIndex: 1)


// hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
var threadsPerGroup = MTLSize(width:32,height:1,depth:1)
var numThreadgroups = MTLSize(width:(myvector.count+31)/32, height:1, depth:1)
computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

computeCommandEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()



// a. Get GPU data
// outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
var data = NSData(bytesNoCopy: outVectorBuffer.contents(),
    length: myvector.count*sizeof(Float), freeWhenDone: false)
// b. prepare Swift array large enough to receive data from GPU
var finalResultArray = [Float](count: myvector.count, repeatedValue: 0)

// c. get data from GPU into Swift array
data.getBytes(&finalResultArray, length:myvector.count * sizeof(Float))

print(finalResultArray[0...9])

// d. YOU'RE ALL SET!



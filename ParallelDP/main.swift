//
//  main.swift
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

import Foundation
import MetalKit

// parameter only needed by the host
let numPeriods = 0 // periods
// parameters needed by both the host and the device
let K = 4  // capacity
let L = 4 // dimension


let salvageValue: Float = 1.5
let holdingCost: Float = 1.11
let orderCost: Float = 1
let disposalCost: Float = 1
let discountRate: Float = 0.95
let price: Float = 10
let max_demand: Float = 1.0

// hardcoded to the following number
// Need to understand more about threadExecutionWidth for optimal config
let threadExecutionWidth = 128

// parameters needs to be transmitted to device
// The order matters
let paramemterVector: [Float] = [
    Float(K),
    Float(L),
    salvageValue,
    holdingCost,
    orderCost,
    disposalCost,
    discountRate,
    price,
    max_demand
]
let distributionVector: [Float] = [
    1.0
]

// basic calcuation of buffer
let numberOfStates = Int(pow(Double(K), Double(L)))
let unitSize = sizeof(Float)
let resultBufferSize = numberOfStates*unitSize

// basic calculation of device related parameter
let numThreadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)

// Initialize Metal
// Get the default device, which is the same as the one monitor is using
var device: MTLDevice! = MTLCreateSystemDefaultDevice()
// In the following, choose the device NOT used by monitor
//let devices: [MTLDevice] = MTLCopyAllDevices()
//for metalDevice: MTLDevice in devices {
//    if metalDevice.headless == true {
//        device = metalDevice
//    }
//}
// exit with an error message if all devices are used by monitor
//if !device.headless {
//    print("no dedicated device found")
//    exit(1)
//}

// Build command queue
var commandQueue: MTLCommandQueue! = device.newCommandQueue()

// Allocate memory on device
let resourceOption = MTLResourceOptions()
var buffer:[MTLBuffer] = [
    device.newBufferWithLength(resultBufferSize, options: resourceOption),
    device.newBufferWithLength(resultBufferSize, options: resourceOption)
]
var actionBuffer:[MTLBuffer] = [
    device.newBufferWithLength(resultBufferSize, options: resourceOption),
    device.newBufferWithLength(resultBufferSize, options: resourceOption)
]
var parameterBuffer:MTLBuffer = device.newBufferWithBytes(paramemterVector, length: unitSize*paramemterVector.count, options: resourceOption)
// put distriburion buffer here
var distributionBuffer:MTLBuffer = device.newBufferWithBytes(paramemterVector, length: unitSize*distributionVector.count, options: resourceOption)

// Get functions from Shaders and add to MTL library
var defaultLibrary: MTLLibrary! = device.newDefaultLibrary()
let initDP = defaultLibrary.newFunctionWithName("initialize")
let pipelineFilterInit = try device.newComputePipelineStateWithFunction(initDP!)
let iterateDP = defaultLibrary.newFunctionWithName("iterate")
let pipelineFilterIterate = try device.newComputePipelineStateWithFunction(iterateDP!)

// Initialize
for l: Int in 1...L {

    let batchSize:uint = uint(pow(Float(K),Float(l-1)))
    let numGroupsBatch = MTLSize(width:(Int(batchSize)+threadExecutionWidth-1)/threadExecutionWidth, height:1, depth:1)
    // print("Batch Size = ", batchSize)
    
    for batchIndex: uint in 1..<uint(K) {
        
        print("Batch Size = ", batchSize, "batchIndex = ", batchIndex)

        let dispatchIterator: [uint] = [batchSize, batchIndex]
        var dispatchBuffer:MTLBuffer = device.newBufferWithBytes(dispatchIterator, length: sizeof(uint)*dispatchIterator.count, options: resourceOption)
        
        var commandBufferInitDP: MTLCommandBuffer! = commandQueue.commandBuffer()
        var encoderInitDP = commandBufferInitDP.computeCommandEncoder()

        encoderInitDP.setComputePipelineState(pipelineFilterInit)

        encoderInitDP.setBuffer(buffer[0], offset: 0, atIndex: 0)
        encoderInitDP.setBuffer(dispatchBuffer, offset: 0, atIndex: 1)
        encoderInitDP.setBuffer(parameterBuffer, offset: 0, atIndex: 2)

        encoderInitDP.dispatchThreadgroups(numGroupsBatch, threadsPerThreadgroup: numThreadsPerGroup)
        encoderInitDP.endEncoding()
        commandBufferInitDP.commit()
        commandBufferInitDP.waitUntilCompleted()

    }
}

// Iterate T periods
// It's import that t starts from 0%2=0, since we start with buffer[0]
for t in 0..<numPeriods {
    
    for l: Int in 0...L {
        var batchSize: uint = 1
        var batchNum: uint = 1
        var batchStart: uint = 0
        if l>0 {
            batchSize = uint(pow(Float(K),Float(l-1)))
            batchStart = 1
            batchNum = uint(K)
        }

        let numGroupsBatch = MTLSize(width:(Int(batchSize)+threadExecutionWidth-1)/threadExecutionWidth, height:1, depth:1)

        for batchIndex: uint in batchStart..<batchNum {

            print("Iterate Batch Size = ", batchSize, "batchIndex = ", batchIndex)
            
            let dispatchIterator: [uint] = [batchSize, batchIndex]
            var dispatchBuffer:MTLBuffer = device.newBufferWithBytes(dispatchIterator, length: sizeof(uint)*dispatchIterator.count, options: resourceOption)
            
            var commandBufferIterateDP: MTLCommandBuffer! = commandQueue.commandBuffer()
            var encoderIterateDP = commandBufferIterateDP.computeCommandEncoder()
            
            encoderIterateDP.setComputePipelineState(pipelineFilterIterate)
            
            encoderIterateDP.setBuffer(buffer[t%2], offset: 0, atIndex: 0)
            encoderIterateDP.setBuffer(buffer[(t+1)%2], offset: 0, atIndex: 1)
            encoderIterateDP.setBuffer(dispatchBuffer, offset: 0, atIndex: 2)
            encoderIterateDP.setBuffer(parameterBuffer, offset: 0, atIndex: 3)
            encoderIterateDP.setBuffer(distributionBuffer, offset: 0, atIndex: 4)
            encoderIterateDP.setBuffer(actionBuffer[0], offset: 0, atIndex: 5)
            encoderIterateDP.setBuffer(actionBuffer[1], offset: 0, atIndex: 6)
            
            encoderIterateDP.dispatchThreadgroups(numGroupsBatch, threadsPerThreadgroup: numThreadsPerGroup)
            encoderIterateDP.endEncoding()
            commandBufferIterateDP.commit()
            commandBufferIterateDP.waitUntilCompleted()
        
        }
    }
}

// Get data fro device
var data = NSData(bytesNoCopy: buffer[numPeriods%2].contents(), length: resultBufferSize, freeWhenDone: false)
var finalResultArray = [Float](count: numberOfStates, repeatedValue: 0)
data.getBytes(&finalResultArray, length:resultBufferSize)

print(finalResultArray[numberOfStates-20..<numberOfStates])



//
//  main.swift
//  ParallelDP
//
//  Created by Jiheng Zhang on 1/1/2016.
//  Copyright © 2016 verse. All rights reserved.
//

import Foundation
import MetalKit

let fileManager = NSFileManager.defaultManager()
// Reading paremeters from plist
// let plistPath = "/Users/zhanghailun/ParallelDP/ParallelDP/parameters.plist"
// let plistPath = "/Users/jz/Developer/ParallelDP/ParallelDP/parameters.plist"

// Alternatively, specify the plist file while running
let path = fileManager.currentDirectoryPath
let args = Process.arguments
if (args.count != 2) {
    print("Please specify the plist file for paremeters!")
    exit(1)
}
let plistPath = path + "/" + args[1]
print(plistPath)
if !fileManager.fileExistsAtPath(plistPath) {
    print("Cannot find plist file!")
    exit(1)
}

// Reading paremeters from plist
let dict = NSDictionary(contentsOfFile: plistPath)
// print(dict)
let numPeriods: Int! = dict!.valueForKey("Periods") as? Int
let L: Int! = dict!.valueForKey("Dimension") as? Int
let K: Int! = dict!.valueForKey("Capacity") as? Int
let holdingCost: Float! = dict!.valueForKey("HoldingCost") as? Float
let salvageValue: Float! = dict!.valueForKey("SalvageValue") as? Float
let orderCost: Float! = dict!.valueForKey("OrderCost") as? Float
let disposalCost: Float! = dict!.valueForKey("DisposalCost") as? Float
let discountRate: Float! = dict!.valueForKey("DiscountRate") as? Float
let price: Float! = dict!.valueForKey("Price") as? Float
let dist: [Float]! = dict!.valueForKey("Distribution") as? [Float]
// hardcoded to the following number
// Need to understand more about threadExecutionWidth for optimal config
let threadExecutionWidth: Int! = dict!.valueForKey("ThreadExecutionWidth") as? Int

print("The complexity is with Capacity", K,"and Dimension", L)

let max_demand: Float = Float(dist.count)
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

// basic calcuation of buffer
let numberOfStates = Int(pow(Double(K), Double(L)))
let unitSize = sizeof(Float)
let resultBufferSize = numberOfStates*unitSize

// basic calculation of device related parameter
let numThreadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)

// Initialize Metal
// Get the default device, which is the same as the one monitor is using
var device: MTLDevice! = MTLCreateSystemDefaultDevice()
//// In the following, choose the device NOT used by monitor
//let devices: [MTLDevice] = MTLCopyAllDevices()
//for metalDevice: MTLDevice in devices {
//    if metalDevice.headless == true {
//        device = metalDevice
//    }
//}
//// exit with an error message if all devices are used by monitor
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
    device.newBufferWithLength(resultBufferSize, options: resourceOption),
    device.newBufferWithLength(resultBufferSize, options: resourceOption), // depletion action
    device.newBufferWithLength(resultBufferSize, options: resourceOption)  // order action
]
var parameterBuffer:MTLBuffer = device.newBufferWithBytes(paramemterVector, length: unitSize*paramemterVector.count, options: resourceOption)
// put distriburion buffer here
var distributionBuffer:MTLBuffer = device.newBufferWithBytes(dist, length: unitSize*dist.count, options: resourceOption)

// Get functions from Shaders and add to MTL library
var DPLibrary: MTLLibrary! = device.newDefaultLibrary()
let initDP = DPLibrary.newFunctionWithName("initialize")
let pipelineFilterInit = try device.newComputePipelineStateWithFunction(initDP!)
let iterateDP = DPLibrary.newFunctionWithName("iterate")
let pipelineFilterIterate = try device.newComputePipelineStateWithFunction(iterateDP!)

var start = NSDate()
// Initialize
for l: Int in 1...L {

    let batchSize:uint = uint(pow(Float(K),Float(l-1)))
    let numGroupsBatch = MTLSize(width:(Int(batchSize)+threadExecutionWidth-1)/threadExecutionWidth, height:1, depth:1)
    // print("Batch Size = ", batchSize)
    
    for batchIndex: uint in 1..<uint(K) {
        
        //print("Batch Size = ", batchSize, "batchIndex = ", batchIndex)

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

            //print("Iterate Batch Size = ", batchSize, "batchIndex = ", batchIndex)
            
            let dispatchIterator: [uint] = [batchSize, batchIndex]
            var dispatchBuffer:MTLBuffer = device.newBufferWithBytes(dispatchIterator, length: sizeof(uint)*dispatchIterator.count, options: resourceOption)
            
            var commandBufferIterateDP: MTLCommandBuffer! = commandQueue.commandBuffer()
            var encoderIterateDP = commandBufferIterateDP.computeCommandEncoder()
            
            encoderIterateDP.setComputePipelineState(pipelineFilterIterate)
            
            encoderIterateDP.setBuffer(buffer[t%2], offset: 0, atIndex: 0)
            encoderIterateDP.setBuffer(buffer[(t+1)%2], offset: 0, atIndex: 1)
            encoderIterateDP.setBuffer(buffer[2], offset: 0, atIndex: 2)
            encoderIterateDP.setBuffer(buffer[3], offset: 0, atIndex: 3)
            encoderIterateDP.setBuffer(dispatchBuffer, offset: 0, atIndex: 4)
            encoderIterateDP.setBuffer(parameterBuffer, offset: 0, atIndex: 5)
            encoderIterateDP.setBuffer(distributionBuffer, offset: 0, atIndex: 6)

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

var data1 = NSData(bytesNoCopy: buffer[2].contents(), length: resultBufferSize, freeWhenDone: false)
var finalResultArray1 = [Float](count: numberOfStates, repeatedValue: 0)
data1.getBytes(&finalResultArray1, length:resultBufferSize)

var data2 = NSData(bytesNoCopy: buffer[3].contents(), length: resultBufferSize, freeWhenDone: false)
var finalResultArray2 = [Float](count: numberOfStates, repeatedValue: 0)
data2.getBytes(&finalResultArray2, length:resultBufferSize)

var end = NSDate()
print("the time elapsed is ", Double(end.timeIntervalSinceDate(start)))
print(finalResultArray[0..<10])
print(finalResultArray[numberOfStates-10..<numberOfStates])
print(finalResultArray1[numberOfStates-10..<numberOfStates])
print(finalResultArray2[numberOfStates-10..<numberOfStates])


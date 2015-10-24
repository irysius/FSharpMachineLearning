namespace NeuralNetwork
open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

// NeuralNetwork.MathHelper
module MathHelper = 
    let sigmoid (x:float) = 
        match x with
        | x when x < -45.0 -> 0.0
        | x when x > 45.0 -> 1.0
        | _ -> 1.0 / (1.0 + Math.Exp(-x))
    
    let tanh (x:float) =
        match x with 
        | x when x < -10.0 -> -1.0
        | x when x > 10.0 -> 1.0
        | _ -> Math.Tanh x

    // basically vector * scalar
    let multiplyLists elements k = 
        elements
        |> List.map (fun element -> element * k)

    // basically vector + vector
    let addLists xs ys =
        List.zip xs ys
        |> List.map (fun (x, y) -> x + y)

    let multiplyAndAdd rows xs =
        List.zip rows xs
        |> List.map (fun (row, x) -> multiplyLists row x)
        |> List.reduce addLists

// NeuralNetwork.Network
type Network(ihWeights: List<List<float>>, ihBiases: List<float>, hoWeights: List<List<float>>, hoBiases: List<float>) = 
    // Begin bulk of the argument checking code
    let numHidden = ihBiases.Length
    let numOutput = hoBiases.Length
    do 
        if numHidden = 0 then invalidArg "ihBiases" "ihBiases cannot be an empty List"
        if numOutput = 0 then invalidArg "hoBiases" "hoBiases cannot be an empty List"

    let ihWeightsLengths = ihWeights
                           |> List.map (fun x -> x.Length)
                           |> Seq.distinct |> Seq.toList
    do
        if ihWeights.Length = 0 then invalidArg "ihWeights" "ihWeights cannot be an empty List"
        if ihWeightsLengths.Length <> 1 then invalidArg "ihWeights" "Inner lists of ihWeights needs to be the same length"
        if ihWeights.Head.Length = 0 then invalidArg "ihWeights" "ihWeights cannot be a List of empty Lists"

    let numInput = ihWeights.Length

    let hoWeightsLengths = hoWeights
                           |> List.map (fun x -> x.Length)
                           |> Seq.distinct |> Seq.toList
    do 
        if hoWeights.Length = 0 then invalidArg "hoWeights" "hoWeights cannot be an empty List"
        if hoWeightsLengths.Length <> 1 then invalidArg "hoWeights" "Inner lists of hoWeights needs to be the same length"
        if hoWeights.Head.Length = 0 then invalidArg "hoWeights" "hoWeights cannot be a List of empty Lists"
        if hoWeights.Length <> numHidden || hoWeights.Head.Length <> numOutput then invalidArg "hoWeights" (String.Format("hoWeights is expected to be {0} rows of {1} items", numHidden, numOutput))
    // End of argument checking code 

    let sumsFor inputs weights biases = 
        MathHelper.multiplyAndAdd weights inputs 
        |> MathHelper.addLists biases

    member this.ComputeOutput(inputs: List<float>) =
        if inputs.Length <> numInput then invalidArg "inputs" (String.Format("inputs is expected to have {0} elements", numInput))
        let ihOutputs = sumsFor inputs ihWeights ihBiases
                        |> List.map MathHelper.sigmoid
        let hoOutputs = sumsFor ihOutputs hoWeights hoBiases
                        |> List.map MathHelper.tanh
        hoOutputs
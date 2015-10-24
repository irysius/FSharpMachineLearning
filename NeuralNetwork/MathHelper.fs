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

    let rowMultiply (matrix:Matrix<float>) (vector:Vector<float>) = 
        Seq.zip (matrix.EnumerateRows()) (vector.Enumerate())
        |> Seq.map (fun (x, y) -> x * y)
        |> DenseMatrix.ofRowSeq

    // Matrix.reduceCols seems to have a bug
    let colCollapse (matrix:Matrix<float>) =
        matrix.EnumerateColumns()
        |> Seq.map (fun x -> x.Sum())
        |> DenseVector.ofSeq

// NeuralNetwork.Network
type Network(ihWeights: Matrix<float>, ihBiases: Vector<float>, hoWeights: Matrix<float>, hoBiases: Vector<float>) = 
    let numHidden = ihBiases.Count
    let numOutput = hoBiases.Count
    let numInput = ihWeights.RowCount
    do
        if ihWeights.ColumnCount <> numHidden then invalidArg "ihWeights" (String.Format("ihWeights is expected to have {0} columns", numHidden))
        if hoWeights.RowCount <> numHidden || hoWeights.ColumnCount <> numOutput then invalidArg "hoWeights" (String.Format("hoWeights is expected to be {0}x{1}", numHidden, numOutput))
    
    let applyWeights weights inputs = 
        MathHelper.rowMultiply weights inputs 
        |> MathHelper.colCollapse

    let applyBiases biases weightedSums =
        weightedSums + biases

    member this.ComputeOutput(inputs: Vector<float>) =
        if inputs.Count <> numInput then invalidArg "inputs" (String.Format("inputs is expected to have {0} elements", numInput))        
        let ihOutputs = inputs
                        |> applyWeights ihWeights
                        |> applyBiases ihBiases
                        |> Vector.map MathHelper.sigmoid
        let hoOutputs = ihOutputs
                        |> applyWeights hoWeights
                        |> applyBiases hoBiases
                        |> Vector.map MathHelper.tanh
        hoOutputs
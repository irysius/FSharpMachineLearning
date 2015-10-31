namespace NeuralNetwork
open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

// NeuralNetwork.MathHelper
module MathHelper = 
    let sigmoid (x:float) = 
        match x with
        | x when x < -45.0 -> 0.0
        | x when x > 45.0 -> 1.0
        | _ -> 1.0 / (1.0 + Math.Exp(-x))

    // derivative of sigmoid
    let dSigmoid (x: float) =
        (sigmoid x) * (1.0 - sigmoid x)
    
    let tanh (x:float) =
        match x with 
        | x when x < -10.0 -> -1.0
        | x when x > 10.0 -> 1.0
        | _ -> Math.Tanh x

    // derivative of tanh
    let dTanh (x:float) =
        (1.0 - tanh x) * (1.0 + tanh x)

    let rowMultiply (matrix:Matrix<float>) (vector:Vector<float>) = 
        Seq.zip (matrix.EnumerateRows()) (vector.Enumerate())
        |> Seq.map (fun (x, y) -> x * y)
        |> DenseMatrix.ofRowSeq

    // Matrix.reduceCols seems to have a bug
    let colCollapse (matrix:Matrix<float>) =
        matrix.EnumerateColumns()
        |> Seq.map (fun x -> x.Sum())
        |> DenseVector.ofSeq

type BackPropagationSetting = {
    learningRate: float
    momentum: float
}

type NetSetting = {
    ihWeights: Matrix<float>
    ihBiases: Vector<float>
    hoWeights: Matrix<float>
    hoBiases: Vector<float>
}

// NeuralNetwork.Net
type Net(setting: NetSetting) = 
    let ihWeights = setting.ihWeights
    let ihBiases = DenseMatrix.OfColumnVectors(setting.ihBiases)
    let hoWeights = setting.hoWeights
    let hoBiases = DenseMatrix.OfColumnVectors(setting.hoBiases)
    let numHidden = setting.ihBiases.Count
    let numOutput = setting.hoBiases.Count
    let numInput = ihWeights.ColumnCount
    do
        if ihWeights.RowCount <> numHidden then invalidArg "ihWeights" (String.Format("ihWeights is expected to be {0}x{1}", numHidden, numInput))
        if hoWeights.ColumnCount <> numHidden || hoWeights.RowCount <> numOutput then invalidArg "hoWeights" (String.Format("hoWeights is expected to be {0}x{1}", numOutput, numHidden))

    member this.ComputeOutput(inputs: Vector<float>) =
        if inputs.Count <> numInput then invalidArg "inputs" (String.Format("inputs is expected to have {0} elements", numInput))
        let inputMatrix = DenseMatrix.OfColumnVectors(inputs)
        let ihOutputs = ihWeights * inputMatrix + ihBiases
                        |> Matrix.map MathHelper.sigmoid
        let hoOutputs = hoWeights * ihOutputs + hoBiases
                        |> Matrix.map MathHelper.tanh
        hoOutputs

    member this.ComputeOutputAndAdjust(inputs: Vector<float>, backPropagate: BackPropagationSetting) =
        0
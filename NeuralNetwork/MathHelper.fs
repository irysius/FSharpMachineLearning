namespace MachineLearning
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

    // derivative of sigmoid, given sigmoid(x)
    let internal dSigmoid (sigmoidx: float) =
        (sigmoidx) * (1.0 - sigmoidx)
    
    let tanh (x:float) =
        match x with 
        | x when x < -20.0 -> -1.0
        | x when x > 20.0 -> 1.0
        | _ -> Math.Tanh x

    // derivative of tanh, given tanh(x)
    let internal dTanh (tanhx:float) =
        (1.0 - tanhx) * (1.0 + tanhx)
    

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

type TrainingData = {
    inputs: Vector<float>
    outputs: Vector<float>
}
type TrainingEndpoint = {
    maxIterations: int
    absoluteError: float
}

// MachineLearning.NeuralNet
type NeuralNet(setting: NetSetting) = 
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
        hoOutputs.Column(0)

    member this.ComputeOutputVerbose(inputs: Vector<float>) = 
        if inputs.Count <> numInput then invalidArg "inputs" (String.Format("inputs is expected to have {0} elements", numInput))
        let inputMatrix = DenseMatrix.OfColumnVectors(inputs)
        let ihOutputs = ihWeights * inputMatrix + ihBiases
                        |> Matrix.map MathHelper.sigmoid
        let hoOutputs = hoWeights * ihOutputs + hoBiases
                        |> Matrix.map MathHelper.tanh
        hoOutputs.Column(0), ihOutputs.Column(0)

module Trainer =
    let trainNet trainingDatum setting (learnRate: float) (momentum: float) trainingEndpoint = 
        let calcOSignals (targets: Vector<float>) (actuals: Vector<float>) = 
            let derivative = actuals
                             |> Vector.map MathHelper.dTanh
            Vector.op_DotMultiply((targets - actuals), derivative)

        let calcHSignals hOutputs (hoWeights: Matrix<float>) (oSignals: Vector<float>) = 
            let y = DenseMatrix.OfRowVectors(oSignals)
            let x = hoWeights
            let sum = y * x
            let row = sum.Row(0)
            let derivative = hOutputs
                             |> Vector.map MathHelper.dSigmoid
            Vector.op_DotMultiply(row, derivative)

        let weightGradients (signals: Vector<float>) (values: Vector<float>) = 
            let x = DenseMatrix.OfColumnVectors(signals)
            let y = DenseMatrix.OfRowVectors(values)
            x * y

        let biasGradients signals = 
            signals
            |> Vector.map (fun x -> x * 1.0)

        let makeSetting ihWeights ihBiases hoWeights hoBiases = 
            {
                ihWeights = ihWeights; 
                ihBiases = ihBiases; 
                hoWeights = hoWeights; 
                hoBiases = hoBiases
            }

        let clearSetting setting = 
            let clone = {
                ihWeights = setting.ihWeights.Clone();
                ihBiases = setting.ihBiases.Clone();
                hoWeights = setting.hoWeights.Clone();
                hoBiases = setting.hoBiases.Clone()
            }
            clone.ihWeights.Clear()
            clone.ihBiases.Clear()
            clone.hoWeights.Clear()
            clone.hoBiases.Clear()
            clone
        
        let updateSetting source gradient delta =
            let updateWeights (weights: Matrix<float>) (weightGradients: Matrix<float>) (prevDelta: Matrix<float>) = 
                let delta = weightGradients * learnRate
                let inertia = prevDelta * momentum
                (weights + delta + inertia), delta

            let updateBiases (biases: Vector<float>) (biasGradients: Vector<float>) (prevDelta: Vector<float>) = 
                let delta = biasGradients * learnRate
                let inertia = prevDelta * momentum
                (biases + delta + inertia), delta

            let (r1, d1) = updateWeights source.ihWeights gradient.ihWeights delta.ihWeights
            let (r2, d2) = updateBiases source.ihBiases gradient.ihBiases delta.ihBiases
            let (r3, d3) = updateWeights source.hoWeights gradient.hoWeights delta.hoWeights
            let (r4, d4) = updateBiases source.hoBiases gradient.hoBiases delta.hoBiases
            ((makeSetting r1 r2 r3 r4), (makeSetting d1 d2 d3 d4))


        let train (setting, delta) trainData = 
            let net = new NeuralNet(setting)
            let (results, ihOutputs) = net.ComputeOutputVerbose(trainData.inputs)
            let oSignals = calcOSignals trainData.outputs results
            let hoWeightGradients = weightGradients oSignals ihOutputs
            let hoBiasGradients = biasGradients oSignals
            let hSignals = calcHSignals ihOutputs setting.hoWeights oSignals
            let ihWeightGradients = weightGradients hSignals trainData.inputs
            let ihBiasGradients = biasGradients hSignals
            let gradient = makeSetting ihWeightGradients ihBiasGradients hoWeightGradients hoBiasGradients
            updateSetting setting gradient delta

        let error setting = 
            let net = new NeuralNet(setting)
            // Mean squared error
            let squaredError = trainingDatum
                               |> List.map (fun x -> (net.ComputeOutput(x.inputs), x.outputs))
                               |> List.map (fun (x, y) -> printfn "%A" x; (x - y) * (x - y))
                               |> List.sum
            squaredError / (float)trainingDatum.Length

        // TODO: Figure out how to do this functionally
        let trainSet setting = 
            let delta = clearSetting setting
            let mutable results = (setting, delta)
            for trainingData in trainingDatum do
                results <- train results trainingData
            let (newSetting, _) = results
            newSetting

        let stopCondition counter setting =
            let e = error setting
            printfn "iteration %d, error: %f" counter e
            let a = (counter >= trainingEndpoint.maxIterations)
            let b = (e < trainingEndpoint.absoluteError)
            (a || b)

        let rec iteration setting counter =
            let newSetting = trainSet setting
            match stopCondition (counter + 1) newSetting with
            | false -> iteration newSetting (counter + 1)
            | true -> newSetting

        let finalSetting = iteration setting 0
        finalSetting
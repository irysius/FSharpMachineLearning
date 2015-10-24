// Learn more about F# at http://fsharp.net. See the 'F# Tutorial' project
// for more guidance on F# programming.
#r @"System.dll"
#r @"System.Core.dll"
#r @"System.Numerics.dll"
#r @"..\packages\MathNet.Numerics.3.8.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.3.8.0\lib\net40\MathNet.Numerics.FSharp.dll"
#r @"bin\Debug\NeuralNetwork.dll"
open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open NeuralNetwork

let inputs = vector [1.0; 2.0; 3.0]

let ihWeights = matrix [[0.1; 0.2; 0.3; 0.4];
                 [0.5; 0.6; 0.7; 0.8];
                 [0.9; 1.0; 1.1; 1.2]]
let ihBiases = vector [-2.0; -6.0; -1.0; -7.0]

let hoWeights = matrix [[1.3; 1.4];
                 [1.5; 1.6];
                 [1.7; 1.8];
                 [1.9; 2.0]]
let hoBiases = vector [-2.5; -5.0]

let network = new Network(ihWeights, ihBiases, hoWeights, hoBiases)
let results = network.ComputeOutput inputs

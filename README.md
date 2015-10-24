# FSharpMachineLearning

## neural_network

### Resources
[James McCaffrey's Post](https://jamesmccaffrey.wordpress.com/2012/06/02/neural-networks/)

[May 2012 Article](https://msdn.microsoft.com/en-us/magazine/hh975375.aspx)

[Suggested Reference](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/preamble.html)

### Notes
I had lots of troubles initially trying to implement McCaffrey's post without resorting to imperative flow control or mutable variables.  

I deviated from his examples by having the net be initialized with the weights and biases to begin with, removing the need to perform a `SetWeights` operation on the neural net afterwards.

Initially the implementation was done with F# Lists, but I decided to use MathNet.Numerics to open up myself to using more mathematical constructs.

## back_propagation
[James McCaffrey's Post](https://jamesmccaffrey.wordpress.com/2012/11/20/coding-neural-network-back-propagation/)

[October 2012 Article](https://msdn.microsoft.com/en-us/magazine/jj658979.aspx)
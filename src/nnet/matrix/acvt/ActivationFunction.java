package nnet.matrix.acvt;

import nnet.matrix.NNetMatrix;

public interface ActivationFunction {
	public static ActivationFunction LINEAR = new ActivationFunctionLinear();
	public static ActivationFunction RELU = new ActivationFunctionReLU();
	public static ActivationFunction TANH = new ActivationFunctionTanh();
	public static ActivationFunction SIGMOID = new ActivationFunctionSigmoid();
	public static ActivationFunction SOFTMAX = new ActivationFunctionSoftMax();
			
	public static ActivationFunction getByType(ActivationType t) {
		var ret = switch (t) {
		case LINEAR -> LINEAR;
		case RELU -> RELU;
		case TANH -> TANH;
		case SIGMOID -> SIGMOID;
		case SOFTMAX -> SOFTMAX;
		};
		
		return ret;
	}
	
	public NNetMatrix f(NNetMatrix m);
	public NNetMatrix grad(NNetMatrix m);
	
}

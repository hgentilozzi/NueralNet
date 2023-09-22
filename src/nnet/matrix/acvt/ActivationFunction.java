package nnet.matrix.acvt;

public interface ActivationFunction {
	public static ActivationFunction LINEAR = new ActivationFunctionLinear();
	public static ActivationFunction RELU = new ActivationFunctionReLU();
	public static ActivationFunction TANH = new ActivationFunctionTanh();
	public static ActivationFunction SIGMOID = new ActivationFunctionSigmoid();
			
	
	public double f(double in);
	public double grad(double in);
}

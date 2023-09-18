package nnet.matrix.acvt;

public class ActivationFunctionReLU implements ActivationFunction {

	public ActivationFunctionReLU() {
	}

	@Override
	public double f(double in) {
		return Math.max(-0.01, in);
	}

	@Override
	public double grad(double in) {
		return (in>=0)? 1 : 0;
	}

}

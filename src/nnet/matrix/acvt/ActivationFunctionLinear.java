package nnet.matrix.acvt;

public class ActivationFunctionLinear implements ActivationFunction {

	public ActivationFunctionLinear() {
	}

	@Override
	public double f(double in) {
		return in;
	}

	@Override
	public double grad(double in) {
		return 1;
	}

}

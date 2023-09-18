package nnet.matrix.acvt;

public class ActivationFunctionTanh implements ActivationFunction {

	public ActivationFunctionTanh() {
	}

	@Override
	public double f(double in) {
		return Math.tanh(in);
	}

	@Override
	public double grad(double in) {
		double fv = f(in);
		return 1-(fv*fv);
	}

}

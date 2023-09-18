package nnet.matrix.acvt;

public class ActivationFunctionSigmoid implements ActivationFunction {

	public ActivationFunctionSigmoid() {
	}

	@Override
	public double f(double in) {
		return (1/(1+Math.exp(-in)));
	}

	@Override
	public double grad(double in) {
		double fv = f(in);
		return fv*(1-fv);
	}

}

package nnet.matrix.acvt;

import nnet.matrix.NNetMatrix;

public class ActivationFunctionSigmoid extends ActivationFunctionLinear {

	public ActivationFunctionSigmoid() {
	}

	@Override
	public NNetMatrix f(NNetMatrix m) {
		return matrixFunc(m,v -> f(v));
	}

	@Override
	public NNetMatrix grad(NNetMatrix m) {
		return matrixFunc(m,v -> grad(v));
	}

	private double f(double in) {
		return (1.0/(1.0+Math.exp(-in)));
	}

	private double grad(double in) {
		double fv = f(in);
		return fv*(1.0-fv);
	}

}

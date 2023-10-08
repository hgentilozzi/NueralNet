package nnet.matrix.acvt;

import nnet.matrix.NNetMatrix;

public class ActivationFunctionTanh extends ActivationFunctionLinear {

	public ActivationFunctionTanh() {
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
		return Math.tanh(in);
	}

	private double grad(double in) {
		double fv = f(in);
		return 1-(fv*fv);
	}

}

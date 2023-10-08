package nnet.matrix.acvt;

import nnet.matrix.NNetMatrix;

public class ActivationFunctionReLU extends ActivationFunctionLinear {
	private double leakValue = 0.0;

	public ActivationFunctionReLU() {
	}
	public double getLeakValue() {
		return leakValue;
	}
	public void setLeakValue(double leakValue) {
		this.leakValue = leakValue;
	}

	@Override
	public NNetMatrix f(NNetMatrix m) {
		return matrixFunc(m,v -> Math.max(leakValue*v, v));
	}

	@Override
	public NNetMatrix grad(NNetMatrix m) {
		return matrixFunc(m,v -> ((v>=0.0)? 1.0 : leakValue));
	}

}

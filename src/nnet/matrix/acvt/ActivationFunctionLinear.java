package nnet.matrix.acvt;

import java.util.function.Function;

import nnet.matrix.NNetMatrix;

public class ActivationFunctionLinear implements ActivationFunction {

	public ActivationFunctionLinear() {
	}

	@Override
	public NNetMatrix f(NNetMatrix m) {
		return m.copy();
	}

	@Override
	public NNetMatrix grad(NNetMatrix m) {
		return m.copy();
	}
	
	/**
	 * Generic update of matrix
	 * @param m - scalar
	 * @return new vector as (this * m)
	 */
	protected NNetMatrix matrixFunc(NNetMatrix m,Function<Double,Double> func) {
		NNetMatrix ret = new NNetMatrix(m.getRows(),m.getCols());
		for (int r=0;r<m.getRows();r++)
			for (int c=0;c<m.getCols();c++)
				ret.set(r,c,func.apply(m.get(r,c)));
		
		return ret;
	}

}

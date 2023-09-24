package nnet.loss;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;

public class NNetLossFuncMeanSquaredError implements NNetLossFunction {

	public NNetLossFuncMeanSquaredError() {
	}

	@Override
	public double getLoss(NNetMatrix actual, NNetMatrix expected) throws MxlibInvalidMatrixOp {
		double sumsqrs = actual.subtract(expected).squareElement().sum();
		return (sumsqrs/expected.getCols());
	}

}

package nnet.loss;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;


public class NNetLossFuncAbsoluteError implements NNetLossFunction {

	public NNetLossFuncAbsoluteError() {
	}

	@Override
	public double getLoss(NNetMatrix actual, NNetMatrix expected) throws MxlibInvalidMatrixOp {
		double sumsqrs = actual.subtract(expected).absoluteValue().sum();
		return (sumsqrs/expected.getCols());
	}

}

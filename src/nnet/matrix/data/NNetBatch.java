package nnet.matrix.data;

import nnet.matrix.NNetMatrix;

public record NNetBatch(int batchSize, NNetMatrix inData, NNetMatrix expectedResults) {

}

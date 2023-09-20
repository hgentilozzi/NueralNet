package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.matrix.data.*;

class TestNNetBatchDataArray {

	@Test
	void testFileNotFound() {
		NNetBatchDataFile ndb = new NNetBatchDataFile("XXXX.XX");
		assertFalse(ndb.validate());
		assertEquals("Failed to initialize. Error=XXXX.XX (The system cannot find the file specified)",
				ndb.getError());
	}

	@Test
	void testFileNOHeader() {
		NNetBatchDataFile ndb = new NNetBatchDataFile("data/junitEmptyDatabase.csv");
		assertFalse(ndb.validate());
		assertEquals("First line of input file does not begin with \"HEADER:\"",
				ndb.getError());
	}


	@Test
	void testBadHeader() {
		NNetBatchDataFile ndb = new NNetBatchDataFile("data/junitBadHeader.csv");
		assertFalse(ndb.validate());
		assertEquals("The header line is ill-formed",
				ndb.getError());
	}

	@Test
	void testGoodFile() throws Exception {
		NNetBatchDataFile ndb = new NNetBatchDataFile("data/junitTrainingDatabase.csv");
		assertTrue(ndb.validate());
		
		assertFalse(ndb.atEof());
		assertEquals(4,ndb.getBatchSize());
		assertEquals(3,ndb.getNumInNodes());
		assertEquals(1,ndb.getNumOutNodes());
		
		NNetBatch b = ndb.nextBatch();
		assertNotNull(b);
		assertEquals(4,b.batchSize());
		assertEquals(60,b.inData().get(0, 0));
		assertEquals(80,b.inData().get(0, 1));
		assertEquals(5,b.inData().get(0, 2));
		assertEquals(82,b.expectedResults().get(0, 0));
		
		assertEquals(4,b.batchSize());
		assertEquals(70,b.inData().get(1, 0));
		assertEquals(75,b.inData().get(1, 1));
		assertEquals(7,b.inData().get(1, 2));
		assertEquals(94,b.expectedResults().get(1, 0));
		
		assertEquals(4,b.batchSize());
		assertEquals(50,b.inData().get(2, 0));
		assertEquals(55,b.inData().get(2, 1));
		assertEquals(10,b.inData().get(2, 2));
		assertEquals(45,b.expectedResults().get(2, 0));
		
		assertEquals(4,b.batchSize());
		assertEquals(40,b.inData().get(3, 0));
		assertEquals(56,b.inData().get(3, 1));
		assertEquals(7,b.inData().get(3, 2));
		assertEquals(43,b.expectedResults().get(3, 0));

		assertTrue(ndb.atEof());

	}

}

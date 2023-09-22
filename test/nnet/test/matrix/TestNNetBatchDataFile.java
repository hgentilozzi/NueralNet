package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.matrix.data.*;

class TestNNetBatchDataFile {

	@Test
	void testFileNotFound() {
		NNetBatchDataFile ndb = new NNetBatchDataFile("XXXX.XX",1,1);
		assertFalse(ndb.validate());
		assertEquals("Failed to initialize. Error=XXXX.XX (The system cannot find the file specified)",
				ndb.getError());
	}

	@Test
	void testGoodFile() throws Exception {
		NNetBatchDataFile ndb = new NNetBatchDataFile("data/junitTrainingDatabase.csv",3,1);
		assertTrue(ndb.validate());
		
		assertFalse(ndb.atEof());
		assertEquals(3,ndb.getNumInNodes());
		assertEquals(1,ndb.getNumOutNodes());
		
		NNetBatch b = ndb.nextBatch(2);
		assertNotNull(b);
		assertEquals(2,b.batchSize());
		assertEquals(60,b.inData().get(0, 0));
		assertEquals(80,b.inData().get(0, 1));
		assertEquals(5,b.inData().get(0, 2));
		assertEquals(82,b.expectedResults().get(0, 0));
		
		assertEquals(70,b.inData().get(1, 0));
		assertEquals(75,b.inData().get(1, 1));
		assertEquals(7,b.inData().get(1, 2));
		assertEquals(94,b.expectedResults().get(1, 0));
		
		b = ndb.nextBatch(2);
		assertNotNull(b);
		assertEquals(2,b.batchSize());
		assertEquals(50,b.inData().get(0, 0));
		assertEquals(55,b.inData().get(0, 1));
		assertEquals(10,b.inData().get(0, 2));
		assertEquals(45,b.expectedResults().get(0, 0));
		
		assertEquals(2,b.batchSize());
		assertEquals(40,b.inData().get(1, 0));
		assertEquals(56,b.inData().get(1, 1));
		assertEquals(7,b.inData().get(1, 2));
		assertEquals(43,b.expectedResults().get(1, 0));

		assertTrue(ndb.atEof());

		b = ndb.nextBatch(2);
		assertNull(b);

	}

}

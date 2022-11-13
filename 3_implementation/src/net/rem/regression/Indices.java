/**
 * REM: REGRESSION MODELS BASED ON EXPECTATION MAXIMIZATION ALGORITHM
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.AlgExtAbstract;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class wraps X indices and Z indices.
 * 
 * @author Loc Nguyen
 * @version 1.0
 * 
 */
public class Indices implements Serializable {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Attribute list.
	 */
	public AttributeList attList = null;
	
	
	/**
	 * Indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * Note, every element of xIndices is an array of objects. In current implementation, only the first object in such array is used, which can be index (the index -1 points to 1 value, not point to profile) or expression.
	 */
	public List<Object[]> xIndices = Util.newList();

	
	/**
	 * Indices for Z data, including -1 such as (-1, z). So z has index 1.
	 * Note, every element of zIndices is an array of objects. In current implementation, only the first object in such array is used, which can be index (the index -1 points to 1 value, not point to profile) or expression.
	 * However, zIndices has 2 elements.
	 */
	public List<Object[]> zIndices = Util.newList();

	
	/**
	 * Default constructor.
	 */
	private Indices() {
		
	}
	
	
	/**
	 * Constructor with attribute list, X indices and Z indices.
	 * @param xIndices  X indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * @param zIndices Z indices for Z data, including -1 such as (-1, z). So z has index 1.
	 */
	public Indices(AttributeList attList, List<Object[]> xIndices, List<Object[]> zIndices) {
		this.attList = attList;
		this.xIndices = xIndices;
		this.zIndices = zIndices;
	}
	
	
	/**
	 * Parsing indices data.
	 * @param cfgIndices configuration indices text.
	 * @param inputSample specified sample.
	 * @return X and Y indices.
	 */
	public static Indices parse(String cfgIndices, Fetcher<Profile> inputSample) {
		return parse(cfgIndices, inputSample, null, null);
	}
	
	
	/**
	 * Parsing indices data.
	 * @param cfgIndices configuration indices text. It can be null, which is for default regression model, y = a0 + a1x1 + a2x2 +... + a(n-1)x(n-1).
	 * @param sample specified sample.
	 * @param xIndicesUsed indicator of used X indices (xIndices).
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, xIndicesUsed[0] is always 0.
	 * @param zIndicesUsed indicator of used Z indices.
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, zIndicesUsed[0] is always 0.
	 * @return X and Y indices.
	 */
	public static Indices parse(String cfgIndices, Fetcher<Profile> sample, int[] xIndicesUsed, int[] zIndicesUsed) {
		if (sample == null) return null;
		Indices indices = new Indices();
		indices.attList = AlgExtAbstract.getSampleAttributeList(sample);
		if (indices.attList == null || indices.attList.size() < 2) return null;

		//Begin parsing indices
		if (indices.xIndices == null) indices.xIndices = Util.newList();
		if (indices.zIndices == null) indices.zIndices = Util.newList();
		if (!parseIndices(cfgIndices, indices.attList.size(), indices.xIndices, indices.zIndices)) //parsing indices
			return null;
		//End parsing indices
		
		//Begin adjusting indices
		if (xIndicesUsed != null && xIndicesUsed.length >= 2)
			indices.xIndices = extractIndicesFromUsed(indices.xIndices, xIndicesUsed);
		if (zIndicesUsed != null && zIndicesUsed.length >= 2)
			indices.zIndices = extractIndicesFromUsed(indices.zIndices, zIndicesUsed);
		if (indices.xIndices == null || indices.xIndices.size() < 2) return null;
		if (indices.zIndices == null || indices.zIndices.size() < 2) return null;
		//End adjusting indices
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[indices.xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		try {
			while (sample.next()) {
				Profile profile = sample.pick(); //profile = (x1, x2,..., x(n-1), z)
				if (profile == null) continue;
				
				double lastValue = RMAbstract.extractNumber(extractResponseValue(profile, indices.attList, indices.zIndices));
				if (Util.isUsed(lastValue)) zExists = zExists || true; 
				
				for (int j = 1; j < indices.xIndices.size(); j++) {
					double value = extractRegressorValue(profile, j, indices.attList, indices.xIndices);
					if (Util.isUsed(value)) xExists[j - 1] = xExists[j - 1] || true;
				}
			}
		}
		catch (Throwable e) {LogUtil.trace(e);}
		finally {
			try {
				if (sample != null) sample.reset();
			} catch (Throwable e) {LogUtil.trace(e);}
		}
		
		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(indices.xIndices.get(0)); //adding -1
		for (int j = 1; j < indices.xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(indices.xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) //Please pay attention here.
			return null;
		indices.xIndices = xIndicesTemp;
		//End checking existence of values.
		
		return indices;
	}
	
    /**
	 * Splitting the specified string into list of indices.
	 * @param cfgIndices specified string.
	 * @return list of indices.
	 */
	public static List<String> splitIndices(String cfgIndices) {
		List<String> txtList = Util.newList();
		if (cfgIndices == null || cfgIndices.isEmpty() || cfgIndices.equals(RM.RM_INDICES_DEFAULT))
			return txtList;
					
		//The pattern is {1, 2}, {3, 4, 5), {5, 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		String regex = "\\}(\\s)*,(\\s)*\\{";
		String[] txtArray = cfgIndices.trim().split(regex);
		for (String txt : txtArray) {
			txt = txt.trim().replaceAll("\\}", "").replaceAll("\\{", "");
			if (!txt.isEmpty())
				txtList.add(txt);
		}
		
		return txtList;
	}
	
	
	/**
	 * Parsing indices of variables. In the most general case, each index is an mathematical expression.
	 * @param cfgIndices input configured indices string.
	 * @param maxVariables input maximum variables.
	 * @param xIndicesOutput output regressors indices. Regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so,
	 * the first element (0) of this indices array is -1 pointing to 1 value. 
	 * @param zIndicesOutput output response indices. Due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @return true if parsing is successful.
	 */
	private static boolean parseIndices(String cfgIndices, int maxVariables, List<Object[]> xIndicesOutput, List<Object[]> zIndicesOutput) {
		xIndicesOutput.clear();
		xIndicesOutput.add(new Object[] {Integer.valueOf(-1)}); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in real data.
		zIndicesOutput.clear();
		zIndicesOutput.add(new Object[] {Integer.valueOf(-1)}); // due to Z = (1, z) and there is no 1 in real data.
		
		//Begin extracting indices from configuration.
		//The pattern is {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		List<String> txtList = splitIndices(cfgIndices);
		
		List<Object[]> indices = Util.newList();
		if (txtList.size() == 1) { //The case: 1, 2, 3, #x4, 5, 5, 6, 5, 6, 7, 8, 9, 10
			List<Object> oneIndices = parseIndex(txtList.get(0), ",");
			for (Object index : oneIndices)
				indices.add(new Object[] {index});
		}
		else if (txtList.size() > 1) { //The case: {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
			for (String txt : txtList) {
				List<Object> oneIndices = parseIndex(txt, ",");
				if (oneIndices.size() == 0)
					continue;
				indices.add(oneIndices.toArray());
			}
		}
		
		if (indices.size() < 2) { //The case: 1, 2, 3, #x4, 5, 5, 6, 5, 6, 7, 8, 9, 10
			for (int j = 0; j < maxVariables - 1; j++)
				xIndicesOutput.add(new Object[] {Integer.valueOf(j)});
			zIndicesOutput.add(new Object[] {Integer.valueOf(maxVariables - 1)}); //The last index is Z index.
		}
		else { //The case: {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
			for (int j = 0; j < indices.size() - 1; j++)
				xIndicesOutput.add(indices.get(j));
			zIndicesOutput.add(indices.get(indices.size() - 1)); //The last index is Z index
		}
		//End extracting indices from configuration
		
		if (zIndicesOutput.size() < 2 || xIndicesOutput.size() < 2)
			return false;
		else
			return true;
	}

	
	/**
	 * Parsing each index.
	 * @param txtIndex text of index.
	 * @param sep separated string.
	 * @return list of indices parsed from text.
	 */
	private static List<Object> parseIndex(String txtIndex, String sep) {
		List<Object> indices = Util.newList();
		if (txtIndex == null || txtIndex.isEmpty())
			return indices;
		List<String> array = TextParserUtil.split(txtIndex, sep, null);
		
		for (String el : array) {
			if (el.contains(RM.VAR_INDEX_SPECIAL_CHAR)) {
				indices.add(el);
				continue;
			}
			
			int index = -1;
			boolean parseSuccess = true;
			try {
				index = Integer.parseInt(el);
			}
			catch (Throwable e) {
				parseSuccess = false;
			}
			if (parseSuccess)
				indices.add(Integer.valueOf(index - 1)); //Index begins 1. Please pay attention to this line.
			else
				indices.add(el);
		}
		
		return indices;
	}

	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile. It can be an array of real values.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter rm_indices.
	 * So the index here is the second index, and of course it is number.
	 * Index starts from 1. So index 0 always indicates to value 1.
	 * @param attList attribute list.
	 * @param xIndices indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * @return value of regressor (X) extracted from specified profile. Note, the returned value is not transformed.
	 */
	public static double extractRegressorValue(Object input, int index, AttributeList attList, List<Object[]> xIndices) {
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return RMAbstract.extractVariableValue(input, null, xIndices, index);
		else
			return RMAbstract.extractVariableValue(input, attList, xIndices, index);
	}

	
	/**
	 * Extracting value of response variable (Z) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile but it can be an array of real values.
	 * @param attList attribute list.
	 * @param zIndices indices for Z data, including -1 such as (-1, z). So z has index 1.
	 * @return value of response variable (Z) extracted from specified profile.
	 */
	public static Object extractResponseValue(Object input, AttributeList attList, List<Object[]> zIndices) {
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return RMAbstract.extractVariableValue(input, null, zIndices, 1);
		else
			return RMAbstract.extractVariableValue(input, attList, zIndices, 1);
	}

	
	/**
	 * Finding object in specified list of indices.
	 * @param indicesList specified list of indices.
	 * @param object specified object.
	 * @return index of specified object in specified list of indices. 
	 */
	public static int findIndex(List<Object[]> indicesList, Object object) {
		for (int i = 0; i < indicesList.size(); i++) {
			Object[] objects = indicesList.get(i);
			for (int j = 0; j < objects.length; j++) {
				if (objects[j].equals(object))
					return i;
			}
		}
		
		return -1;
	}
	
	
	/**
	 * Extracting data as large statistics from sample.
	 * @param sample input sample.
	 * @param attList attribute list.
	 * @param xIndices indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * @param zIndices indices for Z data, including -1 such as (-1, z). So z has index 1.
	 * @return data as large statistics extracted from sample. 
	 */
	public static LargeStatistics extractData(Fetcher<Profile> sample, AttributeList attList, List<Object[]> xIndices, List<Object[]> zIndices) {
		return extractData(sample, attList, xIndices, zIndices, null);
	}
	
	
	/**
	 * Extracting data as large statistics from sample.
	 * @param sample input sample.
	 * @param attList attribute list.
	 * @param xIndices indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * @param zIndices indices for Z data, including -1 such as (-1, z). So z has index 1.
	 * @param transformer transformer for regressors and response. It can be null.
	 * @return data as large statistics extracted from sample. 
	 */
	public static LargeStatistics extractData(Fetcher<Profile> sample, AttributeList attList, List<Object[]> xIndices, List<Object[]> zIndices, Transformer transformer) {
		if (sample == null || attList == null || xIndices == null || zIndices == null) return null;
		if (attList.size() < 2 || xIndices.size() < 2 || zIndices.size() < 2) return null;
		
		List<double[]> xData = Util.newList();
		List<double[]> zData = Util.newList();
		try {
			while (sample.next()) {
				Profile profile = sample.pick(); //profile = (x1, x2,..., x(n-1), z)
				if (profile == null) continue;
				
				double[] xVector = new double[xIndices.size()]; //1, x1, x2,..., x(n-1)
				double[] zVector = new double[2]; //1, z
				xVector[0] = 1.0;
				zVector[0] = 1.0;
				
				double lastValue = RMAbstract.extractNumber(extractResponseValue(profile, attList, zIndices));
				if (!Util.isUsed(lastValue))
					zVector[1] = Constants.UNUSED;
				else
					zVector[1] = transformer != null ? (double)transformer.transformResponse(lastValue, false) : lastValue;
				
				for (int j = 1; j < xIndices.size(); j++) {
					double value = Indices.extractRegressorValue(profile, j, attList, xIndices);
					if (!Util.isUsed(value))
						xVector[j] = Constants.UNUSED;
					else
						xVector[j] = transformer != null ? (double)transformer.transformRegressor(value, false) : value;
				}
				
				zData.add(zVector);
				xData.add(xVector);
			}
		}
		catch (Throwable e) {
			LogUtil.trace(e);
		}
		finally {
			try {
				if (sample != null) sample.reset();
			} catch (Throwable e) {LogUtil.trace(e);}
		}
		//End extracting data
		
		if (xData.size() == 0 || zData.size() == 0)
			return null;
		else
			return new LargeStatistics(xData, zData);
	}
	
	
	
	/**
	 * This class represents used indices.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class Used implements Serializable {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Indicator of used X indices (xIndices).
		 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of xIndices array is -1 pointing to 1 value.
		 * Therefore, xIndicesUsed[0] is always 0.
		 */
		public int[] xIndicesUsed = null;
		
		/**
		 * Indicator of used Z indices (zIndices).
		 * For zIndices, due to Z = (1, z), the first element (0) of zIndices array is -1 pointing to 1 value.
		 * Therefore, zIndicesUsed[0] is always 0.
		 */
		public int[] zIndicesUsed = null;
		
		/**
		 * Constructor with Indicator of used X indices (xIndices) and indicator of used Z indices.
		 * @param xIndicesUsed Indicator of used X indices (xIndices).
		 * @param zIndicesUsed Indicator of used Z indices.
		 */
		public Used(int[] xIndicesUsed, int[] zIndicesUsed) {
			this.xIndicesUsed = xIndicesUsed;
			this.zIndicesUsed = zIndicesUsed;
		}
		
	}


	/**
	 * Extracting used indices from specified information.
	 * @param info specified information.
	 * @return used indices extracted from specified information.
	 */
	public static Used extractUsedIndices(Object...info) {
		if (info == null || info.length == 0) return null;
		for (Object object : info) {
			if (object instanceof Used) return (Used)object;
		}
		
		return null;
	
	
	}
	/**
	 * Creating used sequence of indices.
	 * @param maxIndices the number of elements in X indices (xIndices) or Z indices (zIndices).
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, this parameter maxIndices is the length of xIndices or zIndices.
	 * @param used indicating existent indices. Therefore, used[0] is always 0.
	 * @return bit set as used sequence.
	 */
	public static BitSet usedIndicesToBitset(int maxIndices, int[] used) {
		if (maxIndices <= 0) return null;
		BitSet bsFull = new BitSet(maxIndices); bsFull.set(0, maxIndices);
		if (used == null || used.length < 2 || used[0] != 0) return bsFull;
		
		BitSet bs = new BitSet(maxIndices);
		for (int pos : used) {
			if (pos >= 0 && pos < maxIndices) bs.set(pos);
		}
		
		return bs.cardinality() >= 2 ? bs : bsFull;
	}
	
	
	/**
	 * Converting bit set to used indices.
	 * @param orginalIndices original indices.
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For Z indices (zIndices), due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @param bs bit set pointing to used indices. Therefore, bs[0] is always 1.
	 * @return indices extracted from original indices and used bit set.
	 */
	public static int[] bitsetToUsedIndices(List<Object[]> originalIndices, BitSet bs) {
		if (originalIndices == null || originalIndices.size() < 2) return null;
		
		int[] full = new int[originalIndices.size()];
		for (int i = 0; i < originalIndices.size(); i++) full[i] = i;
		if (bs == null || bs.cardinality() < 2) return full;
		
		List<Integer> list = Util.newList();
		for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) list.add(i);
		
		int[] used = DSUtil.toIntArray(list);
		return used[0] == 0 ? used : full;
	}
	
	
	/**
	 * Extracting used indices.
	 * @param indices original indices.
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @param used indicating existent indices. Therefore, used[0] is always 0.
	 * @return used indices.
	 */
	public static List<Object[]> extractIndicesFromUsed(List<Object[]> indices, int[] used) {
		if (indices == null || indices.size() < 2 || used == null || used.length < 2 || used[0] != 0)
			return indices;
		
		List<Object[]> newIndices = Util.newList();
		for (int idx : used) {
			if (idx >= 0 && idx < indices.size()) newIndices.add(indices.get(idx));
		}

		return newIndices.size() >= 2 ? newIndices : indices;
	}
	
	
}



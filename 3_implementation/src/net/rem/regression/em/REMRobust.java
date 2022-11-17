/**
 * REM: REGRESSION MODELS BASED ON EXPECTATION MAXIMIZATION ALGORITHM
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import java.awt.Component;
import java.io.Serializable;
import java.math.BigInteger;
import java.rmi.RemoteException;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import javax.swing.JOptionPane;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.NoteAlg;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Pair;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.parser.TextParserUtil;
import net.rem.regression.Indices;
import net.rem.regression.LargeStatistics;
import net.rem.regression.MathAdapter;
import net.rem.regression.RMAbstract;
import net.rem.regression.VarWrapper;

/**
 * This class implements the regression model based on robust regressors.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class REMRobust extends REMInclude implements NoteAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Proportion field.
	 */
	public final static String PROPORTION_FIELD = "remro_proportion";
	
	
	/**
	 * Absolute proportion.
	 */
	public final static String PROPORTION_ABSOLUTE = "absolute";

	
	/**
	 * Absolute proportion.
	 */
	public final static String PROPORTION_DECREASE = "decrease";

	
	/**
	 * Absolute proportion.
	 */
	public final static String PROPORTION_INCREASE = "increase";

	
	/**
	 * Absolute proportion.
	 */
	public final static String PROPORTION_DEFAULT = PROPORTION_ABSOLUTE;

	
	/**
	 * Proportion values.
	 */
	public final static String[] PROPORTION_VALUES = {
		PROPORTION_ABSOLUTE,
		PROPORTION_DECREASE,
		PROPORTION_INCREASE,
	};

	
	/**
	 * Optimal mode field.
	 */
	public final static String OPTIMAL_MODE_FIELD = "remro_optimal_mode";
	
	
	/**
	 * Cumulative density function for optimal mode.
	 */
	public final static String OPTIMAL_MODE_CDF = "cdf";

	
	/**
	 * Correlation for optimal mode.
	 */
	public final static String OPTIMAL_MODE_R = "r";

	
	/**
	 * Default optimal mode.
	 */
	public final static String OPTIMAL_MODE_DEFAULT = OPTIMAL_MODE_R;

	
	/**
	 * Proportion values.
	 */
	public final static String[] OPTIMAL_MODE_VALUES = {
		OPTIMAL_MODE_CDF,
		OPTIMAL_MODE_R,
	};

	
	/**
	 * Combination number for evaluation.
	 */
	public final static String COMBINE_NUMBER_FIELD = "remro_combine_number";
	
	
	/**
	 * Default value for combination number for evaluation.
	 */
	public final static int COMBINE_NUMBER_DEFAULT = 0;

	
	/**
	 * Combination number for evaluation in percentage.
	 */
	public final static String COMBINE_PERCENT_FIELD = "remro_combine_percent";
	
	
	/**
	 * Default value for combination number for evaluation in percentage.
	 */
	public final static double COMBINE_PERCENT_DEFAULT = 0.5;

	
	/**
	 * Set of free regressors.
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of xIndices is -1 pointing to 1 value.
	 * Therefore this free X positions pointing to xIndices are from 1. 
	 */
	public final static String FREE_XINDICES_USED_FIELD = "remro_free_xindices";
	
	
	/**
	 * Default set of free regressors. Because X = (1, x1, x2,..., x(n-1), this field 
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of xIndices is -1 pointing to 1 value.
	 * Therefore this free X positions pointing to xIndices are from 1. 
	 */
	public final static String FREE_XINDICES_DEFAULT = ""; //Like "7, 9, 3, 4, 11"

	
	/**
	 * Maximum number of robust regressors.
	 */
	public final static String MAXROGVARS_FIELD = "remro_maxrogvars";
	
	
	/**
	 * Default value for maximum number of regressors. Value 0 indicates entire focused regressors.
	 */
	public final static int MAXREGVARS_DEFAULT = 0;

	
	/**
	 * List of robust regressors.
	 */
	protected List<VarWrapper> robusts = Util.newList();
	
	
	/**
	 * Default constructor.
	 */
	public REMRobust() {
		super();
	}

	
	@Override
	protected REM createREM() {
		return new REMImpl();
	}


	/**
	 * Casting internal model into abstract REM.
	 * @return abstract REM.
	 */
	private REMAbstract rem() {
		return (REMAbstract)rem;
	}
	
	
	@Override
	protected Object learn0() throws RemoteException {
		String optmode = config.getAsString(OPTIMAL_MODE_FIELD);
		if (optmode == null)
			return remro();
		else if (optmode.equals(OPTIMAL_MODE_R))
			return remro();
		else if (optmode.equals(OPTIMAL_MODE_CDF))
			return cdf();
		else
			return remro();
	}


	/**
	 * Learning by REMRO algorithm.
	 * @return regressive parameter.
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unchecked")
	protected Object remro() throws RemoteException {
		robusts.clear();
		REMAbstract rem = rem();
		if (rem == null || this.xIndices == null || this.xIndices.size() < 2) return null;
		
		List<Integer> total = extractRealXIndicesUsed();
		List<Integer> free = extractRealFreeXIndicesUsed();
		List<Integer> focus = Util.newList(); focus.addAll(total); focus.removeAll(free);
		if (focus.size() <= 1) {
			rem.addSetupListener(this);
			rem.setup((Fetcher<Profile>)sample);
			rem.removeSetupListener(this);
			return rem.getParameter();
		}

		int r = config.getAsInt(COMBINE_NUMBER_FIELD);
		if (r <= 0) {
			double rp = config.getAsReal(COMBINE_PERCENT_FIELD);
			if (rp > 0 && rp <= 1) r = (int)(rp * (focus.size()-1));
			r = r >= 1 ? r : focus.size() - 1;
		}
		r = Math.min(r, focus.size() - 1);
		
		int maxRoVars = config.getAsInt(MAXROGVARS_FIELD);
		maxRoVars = maxRoVars <= 0 ? focus.size() : Math.min(maxRoVars, focus.size());
		Map<BitSet, Double> fitMap = Util.newMap();
		List<Pair> roIndices = Util.newList(); //Important list of robust regressor indices with their fitness.

		if (isLearnStarted()) return null;
		
		int maxIteration = focus.size();
		int iteration = 0;
		learnStarted = true;
		while (learnStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			
			List<Integer> doubltful = Util.newList(focus.size());
			doubltful.addAll(focus);
			int varIndex = doubltful.remove(iteration);
			
			CombinationGenerator gen = new CombinationGenerator(doubltful.size(), r);
			double K = 0;
			double fitSum = 0;
			while (gen.hasMore()) {
				int[] comb = gen.getNext();
				int[] xIndicesUsed = new int[2 + free.size() + comb.length];
				xIndicesUsed[0] = 0;
				xIndicesUsed[1] = varIndex;
				for (int i = 0; i < free.size(); i++) xIndicesUsed[2 + i] = free.get(i);
				for (int i = 0; i < comb.length; i++) {
					int used = doubltful.get(comb[i]);
					xIndicesUsed[2 + free.size() + i] = used;
				}
	
				rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
				
				BitSet bs = Indices.usedIndicesToBitset(this.xIndices.size(), xIndicesUsed);
				double fit = Constants.UNUSED;
				if (fitMap.containsKey(bs))
					fit = fitMap.get(bs);
				else {
					//Conditional (local or model) correlation: R(x, y) given model k is R(x, estimated y with model k) multiplied with R(estimated y with model k, y).
					//It means R(x, y | k) = R(x, y estimated with k) * R(y estimated with k, y)
					fit = rem.calcR(1) * rem.calcR();
					
					if (Util.isUsed(fit)) fitMap.put(bs, fit);
				}
				
				if (Util.isUsed(fit)) {
					K += 1;
					fitSum += fit;
				}
			}
			
			if (K != 0) {
				//Averaged conditional (local or model) correlation is average of R(x, y | k) over all k models.
				double fit = fitSum / K;
				
				LargeStatistics data = Indices.extractData((Fetcher<Profile>)sample, this.attList, this.xIndices, this.zIndices, this);
				//Global correlation: R(x, y).
				double globalFit = RMAbstract.calcRRegressorResponse(data, varIndex);
				
				roIndices.add(new Pair(varIndex, fit*globalFit));
			}
			
			iteration ++;

			//Pseudo-code to fire doing setup event.
			fireSetupEvent(new SetupAlgEvent(this, Type.doing, getName(), "Setting up is doing: " + getDescription(), iteration, maxIteration));
			
			synchronized (this) {
				while (learnPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {LogUtil.trace(e);}
				}
			}
			
		}
		
		Collections.sort(roIndices, new Comparator<Pair>() {
			@Override
			public int compare(Pair o1, Pair o2) {
				double fit1 = o1.value(), fit2 = o2.value();
				String prop = getConfig().getAsString(PROPORTION_FIELD);
				
				if (prop.equals(PROPORTION_ABSOLUTE)) {
					fit1 = Math.abs(o1.value());
					fit2 = Math.abs(o2.value());
				}
				else if (prop.equals(PROPORTION_INCREASE)) {
					fit1 = o1.value();
					fit2 = o2.value();
				}
				else if (prop.equals(PROPORTION_DECREASE)) {
					fit1 = o2.value();
					fit2 = o1.value();
				}
					
				if (fit1 > fit2)
					return -1;
				else if (fit1 == fit2)
					return 0;
				else
					return 1;
			}
		});
		roIndices = roIndices.subList(0, maxRoVars);
		
		if (roIndices.size() > 0) {
			int[] xIndicesUsed = new int[1 + roIndices.size() + free.size()];
			xIndicesUsed[0] = 0;
			for (int i = 0; i < roIndices.size(); i++) xIndicesUsed[1 + i] = roIndices.get(i).key();
			for (int i = 0; i < free.size(); i++) xIndicesUsed[1 + roIndices.size() + i] = free.get(i);
			rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
			
			for (int i = 0; i < roIndices.size(); i++) {
				VarWrapper robust = rem.extractRegressor(i + 1);
				robust.setTag(roIndices.get(i).value());
				this.robusts.add(robust);
			}
		}
		else
			rem.setup((Fetcher<Profile>)sample);
		
		synchronized (this) {
			learnStarted = false;
			learnPaused = false;
			
			//Pseudo-code to fire done setup event.
			fireSetupEvent(new SetupAlgEvent(this, Type.done, getName(), "Setting up is done: " + getDescription(), iteration, maxIteration));

			notifyAll();
		}
		
		return rem.getParameter();
	}
	
	
	/**
	 * Learning by cumulative density function (CDF) algorithm.
	 * @return regressive parameter.
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unchecked")
	protected Object cdf() throws RemoteException {
		robusts.clear();
		REMAbstract rem = rem();
		if (rem == null || this.xIndices == null || this.xIndices.size() < 2) return null;
		
		List<Integer> total = extractRealXIndicesUsed();
		List<Integer> free = extractRealFreeXIndicesUsed();
		List<Integer> focus = Util.newList(); focus.addAll(total); focus.removeAll(free);
		if (focus.size() <= 1) {
			rem.addSetupListener(this);
			rem.setup((Fetcher<Profile>)sample);
			rem.removeSetupListener(this);
			return rem.getParameter();
		}

		int r = config.getAsInt(COMBINE_NUMBER_FIELD);
		if (r <= 0) {
			double rp = config.getAsReal(COMBINE_PERCENT_FIELD);
			if (rp > 0 && rp <= 1) r = (int)(rp * (focus.size()-1));
			r = r >= 1 ? r : focus.size() - 1;
		}
		r = Math.min(r, focus.size() - 1);
		
		int maxRoVars = config.getAsInt(MAXROGVARS_FIELD);
		maxRoVars = maxRoVars <= 0 ? focus.size() : Math.min(maxRoVars, focus.size());
		Map<BitSet, double[]> fitMap = Util.newMap();
		List<Pair> roIndices = Util.newList();
		
		if (isLearnStarted()) return null;
		
		int maxIteration = focus.size();
		int iteration = 0;
		learnStarted = true;
		while (learnStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			
			List<Integer> doubltful = Util.newList(focus.size());
			doubltful.addAll(focus);
			int varIndex = doubltful.remove(iteration);
			
			CombinationGenerator gen = new CombinationGenerator(doubltful.size(), r);
			double alphaSum = 0;
			double likelihoodSum = 0;
			while (gen.hasMore()) {
				int[] comb = gen.getNext();
				int[] xIndicesUsed = new int[2 + free.size() + comb.length];
				xIndicesUsed[0] = 0;
				xIndicesUsed[1] = varIndex;
				for (int i = 0; i < free.size(); i++) xIndicesUsed[2 + i] = free.get(i);
				for (int i = 0; i < comb.length; i++) {
					int used = doubltful.get(comb[i]);
					xIndicesUsed[2 + free.size() + i] = used;
				}
	
				rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
				
				BitSet bs = Indices.usedIndicesToBitset(this.xIndices.size(), xIndicesUsed);
				ExchangedParameter parameter = rem.getExchangedParameter();
				double alpha = parameter.getAlpha().get(1);
				double likelihood = Constants.UNUSED;
				if (fitMap.containsKey(bs)) {
					alpha = fitMap.get(bs)[0];
					likelihood = fitMap.get(bs)[1];
				}
				else {
					LargeStatistics stats = rem.getLargeStatistics();
					double variance = parameter.estimateZVariance(stats);
					likelihood = parameter.likelihood(stats, variance, true);
					
					if (Util.isUsed(alpha) && Util.isUsed(likelihood))
						fitMap.put(bs, new double[] {alpha, likelihood});
				}
				
				if (Util.isUsed(alpha) && Util.isUsed(likelihood)) {
					alphaSum += alpha * likelihood;
					likelihoodSum += likelihood;
				}
			}
			
			if (likelihoodSum != 0) {
				double alphaMean = alphaSum / likelihoodSum;
				
				Collection<double[]> fits = fitMap.values();
				double alphaVariance = 0;
				for (double[] fit : fits) {
					double alpha = fit[0];
					double likelihood = fit[1];
					double d = alpha - alphaMean;
					alphaVariance += d*d * likelihood;
				}
				alphaVariance = alphaVariance / likelihoodSum;
				
				double cdf = MathAdapter.normalCDF(0, alphaMean, alphaVariance);
				String prop = getConfig().getAsString(PROPORTION_FIELD);
				if (prop.equals(PROPORTION_ABSOLUTE))
					cdf = Math.max(cdf, 1.0 - cdf);
				else if (prop.equals(PROPORTION_INCREASE))
					cdf = 1.0 - cdf;
				
				roIndices.add(new Pair(varIndex, cdf));
			}
			
			iteration ++;

			//Pseudo-code to fire doing setup event.
			fireSetupEvent(new SetupAlgEvent(this, Type.doing, getName(), "Setting up is doing: " + getDescription(), iteration, maxIteration));
			
			synchronized (this) {
				while (learnPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {LogUtil.trace(e);}
				}
			}
			
		}
		
		Collections.sort(roIndices, new Comparator<Pair>() {
			@Override
			public int compare(Pair o1, Pair o2) {
				double v1 = o1.value(), v2 = o2.value();
					
				if (v1 > v2)
					return -1;
				else if (v1 == v2)
					return 0;
				else
					return 1;
			}
		});
		roIndices = roIndices.subList(0, maxRoVars);
		
		if (roIndices.size() > 0) {
			int[] xIndicesUsed = new int[1 + roIndices.size() + free.size()];
			xIndicesUsed[0] = 0;
			for (int i = 0; i < roIndices.size(); i++) xIndicesUsed[1 + i] = roIndices.get(i).key();
			for (int i = 0; i < free.size(); i++) xIndicesUsed[1 + roIndices.size() + i] = free.get(i);
			rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
			
			for (int i = 0; i < roIndices.size(); i++) {
				VarWrapper robust = rem.extractRegressor(i + 1);
				robust.setTag(roIndices.get(i).value());
				this.robusts.add(robust);
			}
		}
		else
			rem.setup((Fetcher<Profile>)sample);
		
		synchronized (this) {
			learnStarted = false;
			learnPaused = false;
			
			//Pseudo-code to fire done setup event.
			fireSetupEvent(new SetupAlgEvent(this, Type.done, getName(), "Setting up is done: " + getDescription(), iteration, maxIteration));

			notifyAll();
		}
		
		return rem.getParameter();
	}
	
	
	/**
	 * Extract free X positions.
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of xIndices is -1 pointing to 1 value.
	 * Therefore this free X positions pointing to xIndices are from 1. 
	 * @return free X positions.
	 */
	protected List<Integer> extractRealFreeXIndicesUsed() {
		String text = config.getAsString(FREE_XINDICES_USED_FIELD);
		if (text == null) return Util.newList();
		
		List<Integer> xPosListTemp = TextParserUtil.parseListByClass(text, Integer.class, COMBINE_NUMBER_FIELD);
		if (xPosListTemp == null) return Util.newList();
		List<Integer> xPosList = Util.newList(xPosListTemp.size());
		for (int pos : xPosListTemp) {
			if (pos > 0 && !xPosList.contains(pos)) xPosList.add(pos);
		}
		
		return xPosList;
	}
	
	
	/**
	 * Getting X positions.
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of xIndices is -1 pointing to 1 value.
	 * Therefore this X positions pointing to xIndices are from 1. 
	 * @return X positions.
	 */
	protected List<Integer> extractRealXIndicesUsed() {
		if (this.xIndices == null || this.xIndices.size() < 2) return Util.newList();
		List<Integer> xPosList = Util.newList(this.xIndices.size());
		for (int i = 1; i < this.xIndices.size(); i++) xPosList.add(i);
		
		return xPosList;
	}
	
	
	@Override
	public synchronized String getDescription() throws RemoteException {
		String desc = super.getDescription();
		if (robusts.size() == 0) return desc;
		
		StringBuffer buffer = new StringBuffer(desc);
		buffer.append("\nfitnesses: ");
		for (int i = 0; i < robusts.size(); i++) {
			if (i > 0) buffer.append(", ");
			VarWrapper robust = robusts.get(i);
			double fit = RMAbstract.extractNumber(robust.getTag());
			buffer.append(robust.toString() + "=" + MathUtil.format(fit));
		}
		
		return buffer.toString();
	}


	@Override
	public synchronized Inspector getInspector() {
		return RMAbstract.getInspector(this);
	}

	
	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "remro";
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		DataConfig tempConfig = super.createDefaultConfig();
		tempConfig.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		tempConfig.put(COMBINE_NUMBER_FIELD, COMBINE_NUMBER_DEFAULT);
		tempConfig.put(COMBINE_PERCENT_FIELD, COMBINE_PERCENT_DEFAULT);
		tempConfig.put(FREE_XINDICES_USED_FIELD, FREE_XINDICES_DEFAULT);
		tempConfig.put(MAXROGVARS_FIELD, MAXREGVARS_DEFAULT);
		tempConfig.put(PROPORTION_FIELD, PROPORTION_DEFAULT);
		tempConfig.put(OPTIMAL_MODE_FIELD, OPTIMAL_MODE_DEFAULT);
		
		DataConfig config = new DataConfig() {

			/**
			 * Default serial version UID.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public Serializable userEdit(Component comp, String key, Serializable defaultValue) {
				if (key.equals(PROPORTION_FIELD)) {
					String prop = getAsString(key);
					prop = prop == null ? PROPORTION_DEFAULT : prop;
					return (Serializable) JOptionPane.showInputDialog(
							comp, 
							"Please choose one proportion mode", 
							"Choosing proportion mode", 
							JOptionPane.INFORMATION_MESSAGE, 
							null, 
							PROPORTION_VALUES, 
							prop);
					
				}
				else if (key.equals(OPTIMAL_MODE_FIELD)) {
					String optmode = getAsString(key);
					optmode = optmode == null ? OPTIMAL_MODE_DEFAULT : optmode;
					return (Serializable) JOptionPane.showInputDialog(
							comp, 
							"Please choose one optimal mode", 
							"Choosing optimal mode", 
							JOptionPane.INFORMATION_MESSAGE, 
							null, 
							OPTIMAL_MODE_VALUES, 
							optmode);
					
				}
				else
					return tempConfig.userEdit(comp, key, defaultValue);
			}
			
		};
		
		config.putAll(tempConfig);
		return config;
	}


	/**
	 * This class implements the combination generator.
	 * @author Someone in internet
	 * @version 1.0
	 */
	protected static class CombinationGenerator {

		/**
		 * The current combination.
		 */
		private int[] a;
		
		/**
		 * The total number of elements.
		 */
		private int n;
		
		/**
		 * The number of elements for each combination. 
		 */
		private int r;
		
		/**
		 * The remaining number of combinations.
		 */
		private BigInteger numLeft;
		
		/**
		 * The total number of combinations.
		 */
		private BigInteger total;

		/**
		 * Constructor with the total number of elements and the number of elements for each combination.
		 * @param n the total number of elements.
		 * @param r The number of elements for each combination.
		 */
		public CombinationGenerator (int n, int r) {
			if (r > n) {
				throw new IllegalArgumentException ();
			}
			if (n < 1) {
				throw new IllegalArgumentException ();
			}
			this.n = n;
			this.r = r;
			a = new int[r];
			BigInteger nFact = getFactorial (n);
			BigInteger rFact = getFactorial (r);
			BigInteger nminusrFact = getFactorial (n - r);
			total = nFact.divide (rFact.multiply (nminusrFact));
			reset ();
		}

		/**
		 * Resetting the generator.
		 */
		public void reset () {
			for (int i = 0; i < a.length; i++) {
				a[i] = i;
			}
			numLeft = new BigInteger (total.toString ());
		}

		/**
		 * Getting the remaining number of combinations.
		 * @return the remaining number of combinations.
		 */
		public BigInteger getNumLeft () {
			return numLeft;
		}

		/**
		 * Checking whether having more combination.
		 * @return whether having more combination.
		 */
		public boolean hasMore () {
			return numLeft.compareTo (BigInteger.ZERO) == 1;
		}

		/**
		 * Getting the total number of combinations.
		 * @return the total number of combinations.
		 */
		public BigInteger getTotal () {
			return total;
		}

		/**
		 * Getting the factorial of a specified number.
		 * @param n specified number.
		 * @return the factorial of a specified number.
		 */
		private static BigInteger getFactorial(int n) {
			BigInteger fact = BigInteger.ONE;
			for (int i = n; i > 1; i--) {
				fact = fact.multiply (new BigInteger (Integer.toString (i)));
			}
			return fact;
		}

		/**
		 * Getting current combination.
		 * @return current combination.
		 */
		public int[] getNext () {

			if (numLeft.equals (total)) {
				numLeft = numLeft.subtract (BigInteger.ONE);
				return a;
			}

			int i = r - 1;
			while (a[i] == n - r + i) {
				i--;
			}
			a[i] = a[i] + 1;
			for (int j = i + 1; j < r; j++) {
				a[j] = a[i] + j - i;
			}

			numLeft = numLeft.subtract (BigInteger.ONE);
			return a;
		}
		
		/**
		 * Example method.
		 */
		public static void example() {
			String[] elements = {"a", "b", "c", "d", "e", "f", "g"};
			int[] indices;
			CombinationGenerator x = new CombinationGenerator (elements.length, 3);
			StringBuffer combination;
			while (x.hasMore ()) {
				combination = new StringBuffer ();
				indices = x.getNext ();
				for (int i = 0; i < indices.length; i++) {
					combination.append (elements[indices[i]]);
				}
				System.out.println (combination.toString ());
			}
		}
		
	}
	
	
//	@SuppressWarnings("unchecked")
//	@Override
//	protected Object learn0() throws RemoteException {
//		REMAbstract rem = rem();
//		if (rem == null || this.xIndices == null || this.xIndices.size() < 2) return null;
//		
//		List<Integer> total = extractRealXIndicesUsed();
//		List<Integer> free = extractRealFreeXIndicesUsed();
//		List<Integer> focus = Util.newList(); focus.addAll(total); focus.removeAll(free);
//		if (focus.size() <= 1) {
//			rem.addSetupListener(this);
//			rem.setup((Fetcher<Profile>)sample);
//			rem.removeSetupListener(this);
//			return rem.getParameter();
//		}
//
//		int r = config.getAsInt(COMBINE_NUMBER_FIELD);
//		if (r <= 0) {
//			double rp = config.getAsReal(COMBINE_PERCENT_FIELD);
//			if (rp > 0 && rp <= 1) r = (int)(rp * (focus.size()-1));
//			r = r >= 1 ? r : focus.size() - 1;
//		}
//		r = Math.min(r, focus.size() - 1);
//		
//		String optmode = config.getAsString(OPTIMAL_MODE_FIELD);
//		int maxRegVars = config.getAsInt(MAXREGVARS_FIELD);
//		maxRegVars = maxRegVars <= 0 ? focus.size() : Math.min(maxRegVars, focus.size());
//		Map<BitSet, double[]> weightFits = Util.newMap();
//		List<double[]> fits = Util.newList();
//		
//		if (isLearnStarted()) return null;
//		
//		int maxIteration = focus.size();
//		int iteration = 0;
//		learnStarted = true;
//		while (learnStarted && (maxIteration <= 0 || iteration < maxIteration)) {
//			
//			List<Integer> doubltful = Util.newList(focus.size());
//			doubltful.addAll(focus);
//			int varIndex = doubltful.remove(iteration);
//			
//			CombinationGenerator gen = new CombinationGenerator(doubltful.size(), r);
//			double alphaSum = 0;
//			double weightSum = 0;
//			double fitSum = 0;
//			while (gen.hasMore()) {
//				int[] comb = gen.getNext();
//				int[] xIndicesUsed = new int[2 + free.size() + comb.length];
//				xIndicesUsed[0] = 0;
//				xIndicesUsed[1] = varIndex;
//				for (int i = 0; i < free.size(); i++) xIndicesUsed[2 + i] = free.get(i);
//				for (int i = 0; i < comb.length; i++) {
//					int used = doubltful.get(comb[i]);
//					xIndicesUsed[2 + free.size() + i] = used;
//				}
//	
//				rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
//				
//				BitSet bs = Indices.usedIndicesToBitset(this.xIndices.size(), xIndicesUsed);
//				ExchangedParameter parameter = rem.getExchangedParameter();
//				double alpha = parameter.getAlpha().get(1);
//				double weight = Constants.UNUSED;
//				double fit = Constants.UNUSED;
//				if (weightFits.containsKey(bs)) {
//					weight = weightFits.get(bs)[0];
//					fit = weightFits.get(bs)[1];
//				}
//				else {
//					LargeStatistics stats = rem.getLargeStatistics();
//					if (optmode.equals(OPTIMAL_MODE_CDF)) {
//						weight = parameter.likelihood(stats, fit, true);
//						fit = parameter.estimateZVariance(stats);
//					}
//					else {
//						weight = 1.0;
//						
//						//Conditional (local or model) correlation: R(x, y) given model k is R(x, estimated y with model k) multiplied with R(estimated y with model k, y).
//						//It means R(x, y | k) = R(x, y estimated with k) * R(y estimated with k, y)
//						fit = rem.calcR(1) * rem.calcR();
//					}
//					
//					if (Util.isUsed(weight) && Util.isUsed(fit))
//						weightFits.put(bs, new double[] {weight, fit});
//				}
//				
//				if (Util.isUsed(weight) && Util.isUsed(fit)) {
//					weightSum += weight;
//					alphaSum += weight * alpha;
//					fitSum += weight * fit;
//				}
//			}
//			
//			if (weightSum != 0) {
//				if (optmode.equals(OPTIMAL_MODE_CDF)) {
//					double fit = fitSum / weightSum;
//					double alpha = alphaSum / weightSum;
//					double cdfFit = normalCDF(0, alpha, fit);
//					fits.add(new double[] {varIndex, cdfFit});
//				}
//				else {
//					//Averaged conditional (local or model) correlation is average of R(x, y | k) over all k models.
//					double fit = fitSum / weightSum;
//					
//					LargeStatistics data = Indices.extractData((Fetcher<Profile>)sample, this.attList, this.xIndices, this.zIndices, this);
//					//Global correlation: R(x, y).
//					double globalFit = RMAbstract.calcRRegressorResponse(data, varIndex);
//					
//					fits.add(new double[] {varIndex, fit*globalFit});
//				}
//			}
//			
//			iteration ++;
//
//			//Pseudo-code to fire doing setup event.
//			fireSetupEvent(new SetupAlgEvent(this, Type.doing, getName(), "Setting up is doing: " + getDescription(), iteration, maxIteration));
//			
//			synchronized (this) {
//				while (learnPaused) {
//					notifyAll();
//					try {
//						wait();
//					} catch (Exception e) {LogUtil.trace(e);}
//				}
//			}
//			
//		}
//		
//		Collections.sort(fits, new Comparator<double[]>() {
//			@Override
//			public int compare(double[] o1, double[] o2) {
//				double v1 = o1[1], v2 = o2[1];
//				String prop = getConfig().getAsString(PROPORTION_FIELD);
//				if (prop.equals(PROPORTION_ABSOLUTE)) {
//					v1 = Math.abs(o1[1]);
//					v2 = Math.abs(o2[1]);
//				}
//				else if (prop.equals(PROPORTION_INCREASE)) {
//					v1 = o1[1];
//					v2 = o2[1];
//				}
//				else if (prop.equals(PROPORTION_DECREASE)) {
//					v1 = o2[1];
//					v2 = o1[1];
//				}
//					
//				if (v1 > v2)
//					return -1;
//				else if (v1 == v2)
//					return 0;
//				else
//					return 1;
//			}
//		});
//		fits = fits.subList(0, maxRegVars);
//		
//		int[] xIndicesUsed = new int[1+ fits.size() + free.size()];
//		xIndicesUsed[0] = 0;
//		for (int i = 0; i < fits.size(); i++) xIndicesUsed[1 + i] = (int)fits.get(i)[0];
//		for (int i = 0; i < free.size(); i++) xIndicesUsed[1 + fits.size() + i] = free.get(i);
//		rem.setup((Fetcher<Profile>)sample, new Indices.Used(xIndicesUsed, null));
//		
//		synchronized (this) {
//			learnStarted = false;
//			learnPaused = false;
//			
//			//Pseudo-code to fire done setup event.
//			fireSetupEvent(new SetupAlgEvent(this, Type.done, getName(), "Setting up is done: " + getDescription(), iteration, maxIteration));
//
//			notifyAll();
//		}
//		
//		return rem.getParameter();
//	}


}

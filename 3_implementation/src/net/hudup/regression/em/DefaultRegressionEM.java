package net.hudup.regression.em;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.AbstractRegression;

/**
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultRegressionEM extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Inverse mode field name.
	 */
	public final static String REM_INVERSE_MODE_FIELD = "rem_inverse_mode";

	
	/**
	 * Balance mode field name.
	 */
	public final static String REM_BALANCE_MODE_FIELD = "rem_balance_mode";

	
	/**
	 * Default inverse mode field.
	 * The best combination is "REM_INVERSE_MODE_DEFAULT = false" and "REM_BALANCE_MODE_DEFAULT = true" but here is used for faster running. 
	 */
	public final static boolean REM_INVERSE_MODE_DEFAULT = true;

	
	/**
	 * Default balance mode field.
	 * The best combination is "REM_INVERSE_MODE_DEFAULT = false" and "REM_BALANCE_MODE_DEFAULT = true" but here is used for faster running. 
	 */
	public final static boolean REM_BALANCE_MODE_DEFAULT = false;

	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = new ArrayList<>(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = new ArrayList<>(); //1, z
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = new ArrayList<>(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = new ArrayList<>(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public DefaultRegressionEM() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	public synchronized Object learn() throws Exception {
		// TODO Auto-generated method stub
		if (prepareInternalData())
			return super.learn();
		else
			return null;
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		this.xData.clear();
		this.zData.clear();
	}


	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData() throws Exception {
		clearInternalData();
		
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = this.sample.pick();
		}
		this.sample.reset();
		if (profile0 == null)
			return false;
		if (profile0.getAttCount() < 2) //(x1, x2,..., x(n-1), z)
			return false;
		this.attList = profile0.getAttRef();
		AbstractRegression.standardizeAttributeNames(this.attList);
		
		String cfgIndices = null;
		if (this.getConfig().containsKey(AbstractRegression.R_INDICES_FIELD))
			cfgIndices = this.getConfig().getAsString(AbstractRegression.R_INDICES_FIELD).trim();
		if (!AbstractRegression.parseIndices(cfgIndices, profile0.getAttCount(), this.xIndices, this.zIndices)) { //parsing indices
			clearInternalData();
			return false;
		}
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[this.xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = extractResponse(profile);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		this.sample.reset();
		List<Object[]> xIndicesTemp = new ArrayList<>();
		xIndicesTemp.add(this.xIndices.get(0)); //adding -1
		for (int j = 1; j < this.xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(this.xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) {
			clearInternalData();
			return false;
		}
		this.xIndices = xIndicesTemp;
		//End checking existence of values.
		
		//Begin extracting data
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double[] xVector = new double[this.xIndices.size()]; //1, x1, x2,..., x(n-1)
			double[] zVector = new double[2]; //1, z
			xVector[0] = 1;
			zVector[0] = 1;
			
			boolean zExist = false;
			double lastValue = extractResponse(profile);
			if (!Util.isUsed(lastValue))
				zVector[1] = Constants.UNUSED;
			else {
				zVector[1] = (double)transformResponse(lastValue, false);
				zExist = true;
			}
			
			boolean xExist = false;
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (!Util.isUsed(value))
					xVector[j] = Constants.UNUSED;
				else {
					xVector[j] = (double)transformRegressor(value, false);
					xExist = true;
				}
			}
			
			if(zExist || xExist) {
				this.zData.add(zVector);
				this.xData.add(xVector);
			}
		}
		this.sample.close();
		//End extracting data
		
		if (this.xData.size() == 0 || this.zData.size() == 0) {
			clearInternalData();
			return false;
		}
		else
			return true;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
		
		this.xData.clear();
		this.zData.clear();
	}
	
	
	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		double[] alpha = ((ExchangedParameter)currentParameter).getVector();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getMatrix();

		int N = this.zData.size();
		double[] zStatistics = new double[N];
		List<double[]> xStatistics = new ArrayList<>();
		for (int i = 0; i < N; i++) {
			Statistics stat = new Statistics(this.zData.get(i)[1], this.xData.get(i));
			if (!getConfig().getAsBoolean(REM_INVERSE_MODE_FIELD))
				stat = estimate(stat, alpha, betas);
			else
				stat = estimateInverse(stat, alpha, betas);
			
			if (stat == null) //Cannot estimate
				return null;
			else {
				zStatistics[i] = stat.getZStatistic();
				xStatistics.add(stat.getXStatistic());
			}
		}
		
		return new ExchangedParameter(zStatistics, xStatistics);
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		double[] zStatistics = ((ExchangedParameter)currentStatistic).getVector();
		List<double[]> xStatistics = ((ExchangedParameter)currentStatistic).getMatrix();
		if (zStatistics.length == 0 || xStatistics.size() != zStatistics.length)
			return null;
		ExchangedParameter currentParameter = (ExchangedParameter) getCurrentParameter();
		
		int N = zStatistics.length;
		int n = xStatistics.get(0).length; //1, x1, x2,..., x(n-1)
		double[] alpha = calcCoeffs(xStatistics, zStatistics);
		if (alpha == null) {
			if (currentParameter != null)
				alpha = Arrays.copyOf(currentParameter.vector, currentParameter.vector.length);
			else { //Used for initialization so that regression model is always determined.
				alpha = new double[n];
				Arrays.fill(alpha, 0.0);
				for (int i = 0; i < N; i++)
					alpha[0] += zStatistics[i];
				alpha[0] = alpha[0] / (double)N; //constant function z = c
			}
		}
		
		List<double[]> betas = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = new ArrayList<>(N);
			double[] x = new double[N];
			for (int i = 0; i < N; i++) {
				double[] zRow = new double[2];
				Z.add(zRow);
				zRow[0] = 1;
				zRow[1] = zStatistics[i];
				x[i] = xStatistics.get(i)[j];
			}
			double[] beta = calcCoeffs(Z, x);
			if (beta == null) {
				if (currentParameter != null) {
					beta = Arrays.copyOf(currentParameter.matrix.get(j),
						currentParameter.matrix.get(j).length);
				}
				else { //Used for initialization so that regression model is always determined.
					beta = new double[2];
					beta[1] = 0;
					for (int i = 0; i < N; i++)
						beta[0] += xStatistics.get(i)[j];
					beta[0] = beta[0] / (double)N; //constant function x = c
				}
			}
			betas.add(beta);
		}
		
		return new ExchangedParameter(alpha, betas);
	}
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	private Statistics estimate(Statistics stat, double[] alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		int a = 0, b = 0, c = 0;
		List<Integer> U = new ArrayList<>();
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha[j] * xVector[j];
			}
			else {
				a += alpha[j] * betas.get(j)[0];
				c += alpha[j] * betas.get(j)[1];
				U.add(j);
			}
		}
		if (c != 1)
			zStatistic = (a + b) / (1 - c);
		else {
			logger.info("Cannot estimate statistic for Z by expectation, stop estimating for this statistic here because use of other method is wrong.");
			return null;
		}
		
		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
		}
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U);
	}

	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	private Statistics estimateInverse(Statistics stat, double[] alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		List<Integer> U = new ArrayList<>();
		int b = 0;
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha[j] * xVector[j];
				xStatistic[j] = xVector[j]; //existent xij
			}
			else
				U.add(j);
		}

		if (U.size() > 0) {
			//Estimating missing xij (xStatistic) by equation 8, based on current parameter.
			List<double[]> A = new ArrayList<>(U.size());
			double[] y = new double[U.size()];
			
			for (int i = 0; i < U.size(); i++) {
				double[] aRow = new double[U.size()];
				A.add(aRow);
				for (int j = 0; j < U.size(); j++) {
					if (i == j)
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)] - 1;
					else
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)];
				}
				y[i] = -betas.get(U.get(i))[0] - betas.get(U.get(i))[1] * b;
			}
			
			double[] solution = AbstractRegression.solve(A, y); //solve Ax = y
			if (solution != null) {
				for (int j = 0; j < U.size(); j++)
					xStatistic[U.get(j)] = solution[j]; 
			}
			else {
				logger.info("Cannot estimate statistic for X by expectation, stop estimating for this statistic here because use of other method is wrong.");
				return null;
			}
		}
		
		//Estimating missing zi (zStatistic) by equation 4, based on current parameter.
		zStatistic = 0;
		for (int j = 0; j < xStatistic.length; j++) {
			zStatistic += alpha[j] * xStatistic[j];
		}
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U);
	}
	
	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.zData.size();
		int n = this.xData.get(0).length;
		
		double[] alpha0 = new double[n];
		Arrays.fill(alpha0, 0.0);
		List<double[]> betas0 = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			double[] beta0 = new double[2];
			if (j == 0) {
				beta0[0] = 1;
				beta0[1] = 0;
			}
			else {
				beta0[0] = 0;
				beta0[1] = 0;
			}
			betas0.add(beta0);
		}
		ExchangedParameter parameter0 = new ExchangedParameter(alpha0, betas0);
		
		List<Double> zVector = new ArrayList<>();
		List<double[]> xStatistics = new ArrayList<>();
		for (int i = 0; i < N; i++) {
			double zValue = this.zData.get(i)[1];
			if (!Util.isUsed(zValue))
				continue;
			
			double[] xVector = this.xData.get(i);
			boolean missing = false;
			for (int j = 0; j < xVector.length; j++) {
				if (!Util.isUsed(xVector[j])) {
					missing = true;
					break;
				}
			}
			
			if (!missing) {
				zVector.add(zValue);
				xStatistics.add(xVector);
			}
		}
		
		if (zVector.size() == 0)
			return parameter0;
		
		N = zVector.size();
		double[] zStatistics = new double[N];
		for (int i = 0; i < N; i++)
			zStatistics[i] = zVector.get(i);
		ExchangedParameter currentStatistic = new ExchangedParameter(zStatistics, xStatistics);
		try {
			return maximization(currentStatistic);
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		double[] parameter1 = ((ExchangedParameter)currentParameter).getVector();
		double[] parameter2 = ((ExchangedParameter)estimatedParameter).getVector();
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		for (int i = 0; i < parameter1.length; i++) {
			if (parameter1[i] == 0) {
				if (parameter2[i] == 0)
					continue;
				else
					return false;
			}
			else {
				double biasRatio = Math.abs(parameter2[i] - parameter1[i]) / Math.abs(parameter1[i]);
				if (biasRatio > threshold)
					return false;
			}
		}
		
		return true;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.estimatedParameter == null)
			return null;
		double[] alpha = ((ExchangedParameter)this.estimatedParameter).getVector();
		if (alpha == null || alpha.length == 0)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = alpha[0];
		for (int j = 0; j < alpha.length - 1; j++) {
			double value = extractRegressor(profile, j + 1); //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			sum += alpha[j + 1] * (double)transformRegressor(value, false); 
		}
		
		return transformResponse(sum, true);
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultRegressionEM em = new DefaultRegressionEM();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(AbstractRegression.R_INDICES_FIELD, AbstractRegression.R_INDICES_FIELD_DEFAULT);
		config.put(REM_INVERSE_MODE_FIELD, REM_INVERSE_MODE_DEFAULT);
		config.put(REM_BALANCE_MODE_FIELD, REM_BALANCE_MODE_DEFAULT);
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		double[] alpha = ((ExchangedParameter)this.getParameter()).getVector();
		if (alpha.length == 0)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponseName(), false) + " = " + MathUtil.format(alpha[0]));
		for (int j = 0; j < alpha.length - 1; j++) {
			double coeff = alpha[j + 1];
			String variableName = transformRegressor(extractRegressorName(j + 1), false).toString();
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + variableName);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + variableName);
		}
		
		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		double[] array = ((ExchangedParameter)parameter).getVector();
		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < array.length; j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(array[j]));
		}
		
		return buffer.toString();
	}

	
	/**
	 * This class represents the exchanged parameter for this REM algorithm.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected static class ExchangedParameter {
		
		/**
		 * Vector parameter.
		 */
		protected double[] vector = new double[0];
		
		/**
		 * Matrix parameter
		 */
		protected List<double[]> matrix = new ArrayList<>();
		
		/**
		 * Constructor with specified vector and matrix.
		 * @param vector specified vector. It must be not null but can be zero-length.
		 * @param matrix specified matrix. It must be not null but can be zero-length.
		 */
		public ExchangedParameter(double[] vector, List<double[]> matrix) {
			this.vector = vector;
			this.matrix = matrix;
		}
		
		/**
		 * Getting vector parameter.
		 * @return vector parameter.
		 */
		public double[] getVector() {
			return vector;
		}
		
		/**
		 * Getting matrix.
		 * @return matrix.
		 */
		public List<double[]> getMatrix() {
			return matrix;
		}
		
	}
	
	
	/**
	 * This class represents a compound statistic.
	 * @author Loc Nguyen
	 * @version 1.0
	 * 
	 */
	protected static class Statistics {

		/**
		 * Statistic for Z variable.
		 */
		protected double zStatistic = Constants.UNUSED;
		
		/**
		 * Statistic for X variables.
		 */
		protected double[] xStatistic = new double[0];
		
		/**
		 * Constructor with specified statistic for Z variable and statistic for X variables.
		 * @param zStatistic statistic for Z variable.
		 * @param xStatistic statistic for X variables. It must be not null but can be zero-length.
		 */
		public Statistics(double zStatistic, double[] xStatistic) {
			this.zStatistic = zStatistic;
			this.xStatistic = xStatistic;
		}
		
		/**
		 * Getting statistic for Z variable.
		 * @return statistic for Z variable.
		 */
		public double getZStatistic() {
			return zStatistic;
		}
		
		/**
		 * Getting statistic for X variables.
		 * @return statistic for X variables.
		 */
		public double[] getXStatistic() {
			return xStatistic;
		}
		
	}
	
	
	/**
	 * Balancing missing values zi (xStatistic) and xij (xValues).
	 * @param alpha alpha coefficients.
	 * @param betas beta coefficients.
	 * @param zStatistic statistic for Z variable.
	 * @param xStatistic statistic for X variables.
	 * @param U list of missing X values.
	 * @return balanced statistics for Z and X variables. Return null if any error raises.
	 */
	private Statistics balanceStatistics(double[] alpha, List<double[]> betas,
			double zStatistic, double[] xStatistic,
			List<Integer> U) {
		if (!getConfig().getAsBoolean(REM_BALANCE_MODE_FIELD))
			return new Statistics(zStatistic, xStatistic);
		
		double zStatisticNext = Constants.UNUSED;
		double[] xStatisticNext = new double[xStatistic.length];
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		while (t < maxIteration) {
			if (!getConfig().getAsBoolean(REM_INVERSE_MODE_FIELD)) {
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha[j] * xStatistic[j];
				
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatisticNext;
				}
			}
			else {
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
				}
				
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha[j] * xStatisticNext[j];
			}
			
			t++;
			
			//Testing approximation
			boolean approx = true;
			if (zStatistic == 0) {
				if (zStatisticNext == 0)
					approx = true;
				else
					approx = false;
			}
			else
				approx = (Math.abs(zStatisticNext - zStatistic) / Math.abs(zStatistic) <= threshold);
				
			for (int j = 0; j < xStatistic.length; j++) {
				if (xStatistic[j] == 0) {
					if (xStatisticNext[j] == 0)
						approx = approx && true;
					else
						approx = approx && false;
				}
				else
					approx = approx && (Math.abs(xStatisticNext[j] - xStatistic[j]) / Math.abs(xStatistic[j]) <= threshold);
				
				if (!approx) break;
			}
			
			zStatistic = zStatisticNext;
			xStatistic = xStatisticNext;
			zStatisticNext = Constants.UNUSED;
			xStatisticNext = new double[xStatistic.length];
			
			if (approx) break;
		}
		
		return new Statistics(zStatistic, xStatistic);
	}


	/**
	 * Calculating coefficients based on data matrix and data vector.
	 * This method will be improved in the next version.
	 * @param X specified data matrix.
	 * @param x specified data vector.
	 * @return coefficients base on data matrix and data vector. Return null if any error raises.
	 */
	private double[] calcCoeffs(List<double[]> X, double[] x) {
		int N = x.length;
		int n = X.get(0).length;
		
		List<double[]> A = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			double[] aRow = new double[n];
			A.add(aRow);
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int k = 0; k < N; k++) {
					sum += X.get(k)[i] * X.get(k)[j];
				}
				aRow[j] = sum;
			}
		}
		
		double[] b = new double[n];
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int k = 0; k < N; k++)
				sum += X.get(k)[i] * x[k];
			b[i] = sum;
		}
		
		return AbstractRegression.solve(A, b);
	}
	
	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param profile specified profile.
	 * @param index specified indices.
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile profile, int index) {
		// TODO Auto-generated method stub
		return AbstractRegression.defaultExtractVariable(profile, xIndices, index);
	}


	/**
	 * Extracting name of regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified indices.
	 * @return text of regressor (X) extracted.
	 */
	protected String extractRegressorName(int index) {
		// TODO Auto-generated method stub
		return AbstractRegression.defaultExtractVariableName(attList, xIndices, index);
	}

	
	/**
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 */
	@Override
	public double extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		return AbstractRegression.defaultExtractVariable(profile, zIndices, 1);
	}


	/**
	 * Extracting name of response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractResponseName() {
		// TODO Auto-generated method stub
		return AbstractRegression.defaultExtractVariableName(attList, zIndices, 1);
	}


	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		// TODO Auto-generated method stub
		return x;
	}


	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}


}

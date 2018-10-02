package net.hudup.temp;

import static net.hudup.regression.AbstractRM.extractNumber;
import static net.hudup.regression.AbstractRM.extractVariable;
import static net.hudup.regression.AbstractRM.extractVariableValue;
import static net.hudup.regression.AbstractRM.findIndex;
import static net.hudup.regression.AbstractRM.parseIndices;
import static net.hudup.regression.AbstractRM.solve;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.MemFetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.NextUpdate;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.LargeStatistics;
import net.hudup.regression.VarWrapper;
import net.hudup.regression.em.REM;
import net.hudup.regression.em.ui.graph.Graph;

/**
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@NextUpdate
public class RegressionEMImpl extends ExponentialEM implements REM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Balance mode field name.
	 */
	public final static String REM_LOOP_BALANCE_MODE_FIELD = "rem_loop_balance_mode";

	
	/**
	 * Default balance mode field. The new version of this REM algorithm uses one-loop balance. 
	 */
	public final static boolean REM_LOOP_BALANCE_MODE_DEFAULT = false;

	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = Util.newList(); //1, z
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 * This variable is also used as the indicator of successful learning (not null).
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public RegressionEMImpl() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		Object resulted = null;
		if (prepareInternalData(this.sample))
			resulted = super.learn();
		if (resulted == null)
			clearInternalData();

		return resulted;
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
	 * @param inputSample specified sample.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalData();
		
		this.attList = getSampleAttributeList(inputSample);
		if (this.attList.size() < 2)
			return false;

		//Begin parsing indices
		String cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD);
		if (!parseIndices(cfgIndices, this.attList.size(), this.xIndices, this.zIndices)) //parsing indices
			return false;
		//End parsing indices
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[this.xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = extractNumber(extractResponseValue(profile));
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		inputSample.reset();
		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(this.xIndices.get(0)); //adding -1
		for (int j = 1; j < this.xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(this.xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2)
			return false;
		this.xIndices = xIndicesTemp;
		//End checking existence of values.
		
		//Begin extracting data
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double[] xVector = new double[this.xIndices.size()]; //1, x1, x2,..., x(n-1)
			double[] zVector = new double[2]; //1, z
			xVector[0] = 1;
			zVector[0] = 1;
			
			double lastValue = extractNumber(extractResponseValue(profile));
			if (!Util.isUsed(lastValue))
				zVector[1] = Constants.UNUSED;
			else
				zVector[1] = (double)transformResponse(lastValue, false);
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (!Util.isUsed(value))
					xVector[j] = Constants.UNUSED;
				else
					xVector[j] = (double)transformRegressor(value, false);
			}
			
			this.zData.add(zVector);
			this.xData.add(xVector);
		}
		inputSample.reset();
		//End extracting data
		
		if (this.xData.size() == 0 || this.zData.size() == 0)
			return false;
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
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		double[] alpha = ((ExchangedParameter)currentParameter).getVector();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getMatrix();
		Statistics additionalMean = null;
		if (info != null && info.length > 0 && (info[0] instanceof Double))
			additionalMean = (Util.isUsed((double)(info[0])) ? new Statistics((double)(info[0]), null) : null);

		int N = this.zData.size();
		double[] zStatistics = new double[N];
		List<double[]> xStatistics = Util.newList();
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(this.zData.get(i)[1], this.xData.get(i));
			Statistics stat1 = estimate(stat0, alpha, betas, additionalMean);
			Statistics stat2 = estimateInverse(stat0, alpha, betas, additionalMean); //new version of balance process.
			
			Statistics stat = null;
			if (stat1 == null && stat2 == null)
				return null;
			else if (stat1 != null && stat2 != null)
				stat = stat1.mean(stat2); // new balance mode with one loop.
			else if (stat1 != null)
				stat = stat1;
			else
				stat = stat2;
			
			stat = (stat.checkValid() ? stat : null);
			if (stat == null)
				return null;
			zStatistics[i] = stat.getZStatistic();
			xStatistics.add(stat.getXStatistic());
		}
		
		ExchangedParameter newParameter = (ExchangedParameter)((ExchangedParameter)currentParameter).clone();
		newParameter.setVector(alpha);
		newParameter.setMatrix(betas);
		return new ExchangedParameter(zStatistics, xStatistics);
	}

	
	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		double[] zStatistics = ((ExchangedParameter)currentStatistic).getVector();
		List<double[]> xStatistics = ((ExchangedParameter)currentStatistic).getMatrix();
		if (zStatistics.length == 0 || xStatistics.size() != zStatistics.length)
			return null;
		ExchangedParameter currentParameter = (ExchangedParameter) getCurrentParameter();
		
		int N = zStatistics.length;
		int n = xStatistics.get(0).length; //1, x1, x2,..., x(n-1)
		double[] alpha = DSUtil.toDoubleArray(calcCoeffs(xStatistics, zStatistics));
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
		
		List<double[]> betas = Util.newList();
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = Util.newList(N);
			double[] x = new double[N];
			for (int i = 0; i < N; i++) {
				double[] zRow = new double[2];
				Z.add(zRow);
				zRow[0] = 1;
				zRow[1] = zStatistics[i];
				x[i] = xStatistics.get(i)[j];
			}
			double[] beta = DSUtil.toDoubleArray(calcCoeffs(Z, x));
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
		
		ExchangedParameter newParameter = null;
		if (currentParameter == null)
			newParameter = new ExchangedParameter(alpha, betas);
		else {
			newParameter = (ExchangedParameter)currentParameter.clone();
			newParameter.setVector(alpha);
			newParameter.setMatrix(betas);
		}
		
		return newParameter;
	}
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @param additionalMean mean statistics. This parameter can be null.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	private Statistics estimate(Statistics stat, double[] alpha, List<double[]> betas, Statistics additionalMean) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		//Preparing additional means
		double zAdditionalMean = Constants.UNUSED;
		if (additionalMean != null && Util.isUsed(additionalMean.zStatistic) && !Util.isUsed(zValue))
			zAdditionalMean = additionalMean.zStatistic;
		double[] xAdditionalMean = null;
		if (additionalMean != null && additionalMean.xStatistic != null && additionalMean.xStatistic.length == xVector.length) {
			boolean missing = false;
			for (int i = 0; i < additionalMean.xStatistic.length; i++) {
				if (!Util.isUsed(additionalMean.xStatistic[i])) {
					missing = true;
					break;
				}
			}
			if (!missing)
				xAdditionalMean = additionalMean.xStatistic;
			
			missing = false;
			for (int i = 0; i < xVector.length; i++) {
				if (!Util.isUsed(xVector[i])) {
					missing = true;
					break;
				}
			}
			
			if (!missing)
				xAdditionalMean = null;
		}
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else {
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
					if (xAdditionalMean != null)
						xStatistic[j] = (xStatistic[j] + xAdditionalMean[j]) / 2.0;
				}
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		double a = 0, b = 0, c = 0;
		List<Integer> U = Util.newList();
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
		if (c != 1) {
			zStatistic = (a + b) / (1 - c);
		}
		else {
			logger.info("Cannot estimate statistic for Z by expectation (#estimate), stop estimating for this statistic here because use of other method is wrong.");
			return null;
		}
		
		if (Util.isUsed(zAdditionalMean))
			zStatistic = (zStatistic + zAdditionalMean) / 2.0; 

		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else {
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
				if (xAdditionalMean != null)
					xStatistic[j] = (xStatistic[j] + xAdditionalMean[j]) / 2.0;
			}
		}
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U, false);
	}

	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @param additionalMean mean statistics. This parameter can be null.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	private Statistics estimateInverse(Statistics stat, double[] alpha, List<double[]> betas, Statistics additionalMean) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		//Preparing additional means
		double zAdditionalMean = Constants.UNUSED;
		if (additionalMean != null && Util.isUsed(additionalMean.zStatistic) && !Util.isUsed(zValue))
			zAdditionalMean = additionalMean.zStatistic;
		double[] xAdditionalMean = null;
		if (additionalMean != null && additionalMean.xStatistic != null && additionalMean.xStatistic.length == xVector.length) {
			boolean missing = false;
			for (int i = 0; i < additionalMean.xStatistic.length; i++) {
				if (!Util.isUsed(additionalMean.xStatistic[i])) {
					missing = true;
					break;
				}
			}
			if (!missing)
				xAdditionalMean = additionalMean.xStatistic;
			
			missing = false;
			for (int i = 0; i < xVector.length; i++) {
				if (!Util.isUsed(xVector[i])) {
					missing = true;
					break;
				}
			}
			
			if (!missing)
				xAdditionalMean = null;
		}

		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else {
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
					if (xAdditionalMean != null)
						xStatistic[j] = (xStatistic[j] + xAdditionalMean[j]) / 2.0;
				}
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		List<Integer> U = Util.newList();
		double b = 0;
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
			List<double[]> A = Util.newList(U.size());
			List<Double> y = Util.newList(U.size());
			
			for (int i = 0; i < U.size(); i++) {
				double[] aRow = new double[U.size()];
				A.add(aRow);
				for (int j = 0; j < U.size(); j++) {
					if (i == j)
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)] - 1;
					else
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)];
				}
				double yi = -betas.get(U.get(i))[0] - betas.get(U.get(i))[1] * b;
				y.add(yi);
			}
			
			List<Double> solution = solve(A, y); //solve Ax = y
			if (solution != null) {
				for (int j = 0; j < U.size(); j++) {
					int k = U.get(j);
					xStatistic[k] = solution.get(j);
					
					if (xAdditionalMean != null)
						xStatistic[k] = (xStatistic[k] + xAdditionalMean[k]) / 2.0;
				}
			}
			else {
				logger.info("Cannot estimate statistic for X by expectation (#estimateInverse), stop estimating for this statistic here because use of other method is wrong.");
				return null;
			}
		}
		
		//Estimating missing zi (zStatistic) by equation 4, based on current parameter.
		zStatistic = 0;
		for (int j = 0; j < xStatistic.length; j++) {
			zStatistic += alpha[j] * xStatistic[j];
		}
		if (Util.isUsed(zAdditionalMean))
			zStatistic = (zStatistic + zAdditionalMean) / 2.0; 
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U, true);
	}
	
	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.zData.size();
		int n = this.xData.get(0).length;
		
		double[] alpha0 = new double[n];
		Arrays.fill(alpha0, 0.0);
		List<double[]> betas0 = Util.newList();
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
		
		List<Double> zVector = Util.newList();
		List<double[]> xStatistics = Util.newList();
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
			ExchangedParameter parameter = (ExchangedParameter) maximization(currentStatistic);
			return (parameter != null ? parameter : parameter0); 
		}
		catch (Throwable e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		ExchangedParameter parameter1 = ((ExchangedParameter)currentParameter);
		ExchangedParameter parameter2 = ((ExchangedParameter)estimatedParameter);
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		double[] alpha1 = parameter1.getVector();
		double[] alpha2 = parameter2.getVector();
		if (alpha1.length != alpha2.length)
			return false;
		for (int i = 0; i < alpha1.length; i++) {
			if (Math.abs(alpha2[i] - alpha1[i]) > threshold * Math.abs(alpha1[i]))
				return false;
		}
		
//		List<double[]> betas1 = parameter1.getMatrix();
//		List<double[]> betas2 = parameter2.getMatrix();
//		if (betas1.size() != betas2.size())
//			return false;
//		for (int i = 0; i < betas1.size(); i++) {
//			double[]  beta1 = betas1.get(i);
//			double[]  beta2 = betas2.get(i);
//			if (beta1.length != beta2.length)
//				return false;
//			
//			for (int j = 0; j < beta1.length; j++) {
//				if (Math.abs(beta2[j] - beta1[j]) > threshold * Math.abs(beta1[j]))
//					return false;
//			}
//		}

		double c1 = parameter1.getCompProb();
		double c2 = parameter2.getCompProb();
		if (Util.isUsed(c1) && Util.isUsed(c2)) {
			if (Math.abs(c2 - c1) > threshold * Math.abs(c1))
				return false;
		}
		else if (Util.isUsed(c1) || Util.isUsed(c2))
			return false;
		
		double mean1 = parameter1.getMean();
		double mean2 = parameter2.getMean();
		if (Util.isUsed(mean1) && Util.isUsed(mean2)) {
			if (Math.abs(mean2 - mean1) > threshold * Math.abs(mean1))
				return false;
		}
		else if (Util.isUsed(mean1) || Util.isUsed(mean2))
			return false;

		double variance1 = parameter1.getVariance();
		double variance2 = parameter2.getVariance();
		if (Util.isUsed(variance1) && Util.isUsed(variance2)) {
			if (Math.abs(variance2 - variance1) > threshold * Math.abs(variance1))
				return false;
		}
		else if (Util.isUsed(variance1) || Util.isUsed(variance2))
			return false;

		return true;
	}

	
	@Override
	public LargeStatistics getLargeStatistics() {
		// TODO Auto-generated method stub
		return null;
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
			if (!Util.isUsed(value))
				return null;
			sum += alpha[j + 1] * (double)transformRegressor(value, false); 
		}
		
		return transformResponse(sum, true);
	}
	
	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 */
	public Object executeIntel(Object...input) {
		return execute(input);
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem.temp";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		RegressionEMImpl em = new RegressionEMImpl();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(REM_LOOP_BALANCE_MODE_FIELD, REM_LOOP_BALANCE_MODE_DEFAULT); //The new version of this REM algorithm uses one-loop balance.
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		ExchangedParameter exParameter = ((ExchangedParameter)this.getParameter());
		double[] alpha = exParameter.getVector();
		if (alpha.length == 0)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponseName(), false) + " = " + MathUtil.format(alpha[0]));
		for (int j = 0; j < alpha.length - 1; j++) {
			double coeff = alpha[j + 1];
			String regressorExpr = "(" + transformRegressor(extractRegressorName(j + 1), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
		}
		
		double c = exParameter.getCompProb();
		double mean = exParameter.getMean();
		double variance = exParameter.getMean();
		if (Util.isUsed(c) || Util.isUsed(mean) || Util.isUsed(variance))
			buffer.append(": ");
		
		if (Util.isUsed(c))
			buffer.append("c=" + MathUtil.format(c));
		if (Util.isUsed(mean))
			buffer.append(", mean=" + MathUtil.format(mean));
		if (Util.isUsed(variance))
			buffer.append(", variance=" + MathUtil.format(variance));

		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		
		ExchangedParameter exParameter = ((ExchangedParameter)parameter);
		double[] alpha = exParameter.getVector();
		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < alpha.length; j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(alpha[j]));
		}
		
		double c = exParameter.getCompProb();
		double mean = exParameter.getMean();
		double variance = exParameter.getMean();
		if (Util.isUsed(c) || Util.isUsed(mean) || Util.isUsed(variance))
			buffer.append(": ");
		
		if (Util.isUsed(c))
			buffer.append("c=" + c);
		if (Util.isUsed(mean))
			buffer.append(", mean=" + mean);
		if (Util.isUsed(variance))
			buffer.append(", variance=" + variance);
			
		return buffer.toString();
	}

	
	/**
	 * Balancing missing values zi (xStatistic) and xij (xValues).
	 * @param alpha alpha coefficients.
	 * @param betas beta coefficients.
	 * @param zStatistic statistic for Z variable.
	 * @param xStatistic statistic for X variables.
	 * @param U list of missing X values.
	 * @param inverse if true, this is inverse mode.
	 * @return balanced statistics for Z and X variables. Return null if any error raises.
	 */
	private Statistics balanceStatistics(double[] alpha, List<double[]> betas,
			double zStatistic, double[] xStatistic,
			List<Integer> U, boolean inverse) {
		if (!getConfig().getAsBoolean(REM_LOOP_BALANCE_MODE_FIELD))
			return new Statistics(zStatistic, xStatistic);
		/*
		 * Balance process does not use additional means because it only uses regression model.
		 */
		
		double zStatisticNext = Constants.UNUSED;
		double[] xStatisticNext = new double[xStatistic.length];
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		while (t < maxIteration) {
			if (!inverse) {
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
			boolean approx = (Math.abs(zStatisticNext - zStatistic) <= threshold * Math.abs(zStatistic));
			for (int j = 0; j < xStatistic.length; j++) {
				approx = approx && 
					(Math.abs(xStatisticNext[j] - xStatistic[j]) <= threshold * Math.abs(xStatistic[j]));
				if (!approx) break;
			}
			
			zStatistic = zStatisticNext;
			xStatistic = xStatisticNext;
			zStatisticNext = Constants.UNUSED;
			xStatisticNext = new double[xStatistic.length];
			
			if (approx) break;
		} //If the likelihood function is too acute, the loop can be infinite.
		
		return new Statistics(zStatistic, xStatistic);
	}


	/**
	 * Calculating coefficients based on data matrix and data vector.
	 * This method will be improved in the next version.
	 * @param X specified data matrix.
	 * @param x specified data vector.
	 * @return coefficients base on data matrix and data vector. Return null if any error raises.
	 */
	public static List<Double> calcCoeffs(List<double[]> X, double[] x) {
		int N = x.length;
		int n = X.get(0).length;
		
		List<double[]> A = Util.newList(n);
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
		
		List<Double> b = Util.newList(n);
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int k = 0; k < N; k++)
				sum += X.get(k)[i] * x[k];
			b.add(sum);
		}
		
		return solve(A, b);
	}
	
	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile.
	 * @param index specified indices.
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile input, int index) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, xIndices, index);
		else
			return extractVariableValue(input, attList, xIndices, index);
	}


	/**
	 * Extracting name of regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified indices.
	 * @return text of regressor (X) extracted.
	 */
	protected String extractRegressorName(int index) {
		// TODO Auto-generated method stub
		return extractVariable(attList, xIndices, index).toString();
	}

	
	@Override
	public List<Double> extractRegressorStatistic(VarWrapper regressor) {
		// TODO Auto-generated method stub
		return null;
	}


	/**
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 */
	@Override
	public Object extractResponseValue(Object input) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, zIndices, 1);
		else
			return extractVariableValue(input, attList, zIndices, 1);
	}


	/**
	 * Extracting name of response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return text of response variable (Z) extracted.
	 */
	public String extractResponseName() {
		// TODO Auto-generated method stub
		return extractVariable(attList, zIndices, 1).toString();
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


	@Override
	public Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}


	/**
	 * Calculating residual mean from specified sample.
	 * @param inputSample specified sample.
	 * @return residual mean from specified sample.
	 */
	public double residualMean(Fetcher<Profile> inputSample) {
		double ss = 0;
		int count = 0;
		
		try {
			while (inputSample.next()) {
				Profile profile = inputSample.pick();
				if (profile == null)
					continue;
				
				double zValue = extractNumber(extractResponseValue(profile));
				double executedValue = extractNumber(execute(profile));
				if (Util.isUsed(zValue) && Util.isUsed(executedValue)) {
					double d = executedValue - zValue;
					ss += d*d;
					count++;
				}
			}
			inputSample.reset();
			
			if (count == 0)
				return Constants.UNUSED;
			else
				return ss / (double)count;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return Constants.UNUSED;
	}
	
	
	/**
	 * Estimating given sample.
	 * @param inputSample given sample.
	 * @param additionalMean additional mean.
	 * @return estimated sample.
	 */
	public Fetcher<Profile> estimate(Fetcher<Profile> inputSample, Statistics additionalMean) {
		// TODO Auto-generated method stub
		double[] alpha = ((ExchangedParameter)getParameter()).getVector();
		List<double[]> betas = ((ExchangedParameter)getParameter()).getMatrix();
		RegressionEMImpl em = new RegressionEMImpl();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		try {
			if (!em.prepareInternalData(inputSample))
				return null;
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
		int N = em.zData.size();
		double[] zStatistics = new double[N];
		List<double[]> xStatistics = Util.newList();
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(this.zData.get(i)[1], this.xData.get(i));
			Statistics stat1 = estimate(stat0, alpha, betas, additionalMean);
			Statistics stat2 = estimateInverse(stat0, alpha, betas, additionalMean); //new version of balance process.
			
			Statistics stat = null;
			if (stat1 == null && stat2 == null)
				return null;
			else if (stat1 != null && stat2 != null)
				stat = stat1.mean(stat2); // new balance mode with one loop.
			else if (stat1 != null)
				stat = stat1;
			else
				stat = stat2;
			
			stat = (stat.checkValid() ? stat : null);
			stat = (stat == null ? stat0 : stat);
			zStatistics[i] = stat.getZStatistic();
			xStatistics.add(stat.getXStatistic());
		}
		
		AttributeList attRef = getSampleAttributeList(inputSample);
		List<Profile> profiles = Util.newList();
		for (int i = 0; i < N; i++) {
			Profile profile = new Profile(attRef);
			double[] xvector = xStatistics.get(i);
			double z = zStatistics[i];
			profile.setValue(attRef.size() - 1, z);
			
			for (int j = 0; j < attRef.size() - 1; j++) {
				int foundX = findIndex(em.xIndices, j);
				if (foundX >= 0)
					profile.setValue(j, xvector[foundX]);
			}
			profiles.add(profile);
		}
		em.clearInternalData();

		return new MemFetcher<>(profiles);
	}


	@Override
	public double calcVariance() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double calcR() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double executeByXStatistic(double[] xStatistic) {
		// TODO Auto-generated method stub
		return Constants.UNUSED;
	}


	@Override
	public Graph createRegressorGraph(VarWrapper regressor) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Graph createResponseGraph() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Graph createErrorGraph() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<Graph> createResponseRalatedGraphs() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public double[] calcError() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public VarWrapper extractRegressor(int index) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public double extractRegressorValue(Object input, int index) {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public VarWrapper extractResponse() {
		// TODO Auto-generated method stub
		return null;
	}


}


/**
 * This class represents the exchanged parameter for this REM algorithm.
 * @author Loc Nguyen
 * @version 1.0
 */
class ExchangedParameter {

	
	/**
	 * Vector parameter.
	 */
	protected double[] vector = null; //As usual, it is alpha coefficients or Z statistics.
	
	
	/**
	 * Matrix parameter
	 */
	protected List<double[]> matrix = null; //As usual, it is beta coefficients or X statistics. 
	
	
	/**
	 * Probability associated with this component. This variable is not used for normal regression model.
	 */
	protected double c = Constants.UNUSED;
	
	
	/**
	 * Mean associated with this component. This variable is not used for normal regression model.
	 */
	protected double mean = Constants.UNUSED;
	
	
	/**
	 * Variance associated with this component. This variable is not used for normal regression model.
	 */
	protected double variance = Constants.UNUSED;
	
	
	/**
	 * Default constructor.
	 */
	private ExchangedParameter() {
		
	}
	
	
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
	 * Constructor with specified vector, matrix, component probability, mean, and variance.
	 * @param vector specified vector. It must be not null but can be zero-length.
	 * @param matrix specified matrix. It must be not null but can be zero-length.
	 */
	public ExchangedParameter(double[] vector, List<double[]> matrix, double c, double mean, double variance) {
		this.vector = vector;
		this.matrix = matrix;
		this.c = c;
		this.mean = mean;
		this.variance = variance;
	}

	
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		ExchangedParameter newParameter = new ExchangedParameter();
		newParameter.vector = Arrays.copyOf(this.vector, this.vector.length);
		newParameter.matrix = Util.newList();
		for (double[] array : this.matrix) {
			newParameter.matrix.add(Arrays.copyOf(array, array.length));
		}
		
		newParameter.c = this.c;
		newParameter.mean = this.mean;
		newParameter.variance = this.variance;
		
		return newParameter;
	}

	/**
	 * Getting vector parameter.
	 * @return vector parameter.
	 */
	public double[] getVector() {
		return vector;
	}
	
	
	/**
	 * Setting vector parameter.
	 * @param vector specified parameter.
	 */
	public void setVector(double[] vector) {
		this.vector = vector;
	}

	
	/**
	 * Getting matrix.
	 * @return matrix.
	 */
	public List<double[]> getMatrix() {
		return matrix;
	}
	
	
	/**
	 * Setting matrix parameter.
	 * @param matrix specified parameter.
	 */
	public void setMatrix(List<double[]> matrix) {
		this.matrix = matrix;
	}

	
	/**
	 * Getting associated probability of component.
	 * @return associated probability of component.
	 */
	public double getCompProb() {
		return c;
	}
	
	
	/**
	 * Setting component probability.
	 * @param c specified component probability.
	 */
	public void setCompProb(double c) {
		this.c = c;
	}

	
	/**
	 * Getting associated probability of component.
	 * @return associated probability of component.
	 */
	public double getMean() {
		return mean;
	}
	
	
	/**
	 * Setting associated mean.
	 * @param mean specified mean.
	 */
	public void setMean(double mean) {
		this.mean = mean;
	}

	
	/**
	 * Getting associated variance of component.
	 * @return associated variance of component.
	 */
	public double getVariance() {
		return variance;
	}
	
	
	/**
	 * Setting associated variance.
	 * @param mean specified variance.
	 */
	public void setVariance(double variance) {
		this.variance = variance;
	}


	/**
	 * Getting component probabilities from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return component probabilities from specified list of parameters.
	 */
	public static double[] getCompProbs(ExchangedParameter...parameters) {
		double[] array = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			array[i] = parameters[i].getCompProb();
		}
		
		return array;
	}
	
	
	/**
	 * Getting means from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return component probabilities from specified list of parameters.
	 */
	public static double[] getMeans(ExchangedParameter...parameters) {
		double[] array = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			array[i] = parameters[i].getMean();
		}
		
		return array;
	}

	
	/**
	 * Getting component variances from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return component variances from specified list of parameters.
	 */
	public static double[] getVariances(ExchangedParameter...parameters) {
		double[] array = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			array[i] = parameters[i].getVariance();
		}
		
		return array;
	}

	
	/**
	 * Calculating the condition probabilities of the specified parameters given response value (Z).
	 * @param z given response value (Z).
	 * @param parameters arrays of parameters.
	 * @return condition probabilities of the specified parameters given response value (Z).
	 */
	public static double[] compCondProbs(double z, ExchangedParameter...parameters) {
		double[] cs = getCompProbs(parameters); 
		double[] means = getMeans(parameters);
		double[] variances = getVariances(parameters);
		
		return compCondProbs(z, cs, means, variances);
	}
	
	
	/**
	 * Calculating the condition probabilities of the specified components given response value (Z), mean, and variance.
	 * @param z given response value (Z).
	 * @param cs array of probabilities of components.
	 * @param means given means.
	 * @param variances given variances.
	 * @return condition probabilities of the specified components given response value (Z), means, and variances. 
	 */
	protected static double[] compCondProbs(double z, double[] cs, double[] means, double[] variances) {
		double[] numerators = new double[cs.length];
		double denominator = 0;
		
		for (int i = 0; i < cs.length; i++) {
			double p = p(z, means[i], variances[i]);
			double value = cs[i] * p;
			
			denominator += value;
			numerators[i] = value;
		}
		
		double[] condProbs = new double[cs.length];
		for (int i = 0; i < cs.length; i++) {
			if (denominator == 0)
				condProbs[i] = 1.0 / cs.length;
			else
				condProbs[i] = numerators[i] / denominator;
		}
		
		return condProbs; 
	}

	
	/**
	 * Calculating the condition probability of the specified component given response value (Z), means, and variances.
	 * @param k specified component.
	 * @param z given response value (Z).
	 * @param cs array of probabilities of components.
	 * @param means given means.
	 * @param variances given variances.
	 * @return condition probability of the specified component given response value (Z), means, and variances. 
	 */
	protected static double compCondProb(int k, double z, double[] cs, double[] means, double[] variances) {
		double numerator = 0;
		double denominator = 0;
		
		for (int i = 0; i < cs.length; i++) {
			double p = p(z, means[i], variances[i]);
			double value = cs[i] * p;
			
			denominator += value;
			if (i == k)
				numerator = value;
		}
		
		return numerator / denominator; 
	}
	
	
	/**
	 * Evaluating the normal probability density function with specified mean and variance.
	 * @param z specified response value z.
	 * @param X specified vector of regressor values X.
	 * @param alpha specified coefficients.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	protected static double p(double z, double[] X, double[] alpha, double variance) {
		double mean = mean(X, alpha);
		return p(z, mean, variance);
	}
	
	
	/**
	 * Evaluating the normal probability density function with specified mean and variance.
	 * @param z specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	protected static double p(double z, double mean, double variance) {
		double d = z - mean;
		return (1.0 / Math.sqrt(2*Math.PI*variance)) * Math.exp(-(d*d) / (2*variance));
	}
	
	
	/**
	 * Evaluating mean from regressors and coefficients.
	 * @param X specified regressors.
	 * @param alpha specified coefficients.
	 * @return mean from regressors and coefficients.
	 */
	protected static double mean(double[] X, double[] alpha) {
		double mean = 0;
		for (int i = 0; i < alpha.length; i++)
			mean += alpha[i] * X[i];
		
		return mean;
	}


}


/**
 * This class represents a compound statistic.
 * @author Loc Nguyen
 * @version 1.0
 * 
 */
class Statistics {

	
	/**
	 * Statistic for Z variable.
	 */
	protected double zStatistic = Constants.UNUSED;
	
	
	/**
	 * Statistic for X variables.
	 */
	protected double[] xStatistic = null;
	
	
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
	
	
	/**
	 * Calculating mean of this statistics and other statistics.
	 * @param other other statistics.
	 * @return mean of this statistics and other statistics.
	 */
	public Statistics mean(Statistics other) {
		double zStatistic = (this.zStatistic + other.zStatistic) / 2.0;
		double[] xStatistic = new double[this.xStatistic.length];
		for (int i = 0; i < xStatistic.length; i++)
			xStatistic[i] = (this.xStatistic[i] + other.xStatistic[i]) / 2.0;
		
		return new Statistics(zStatistic, xStatistic);
	}
	
	
	/**
	 * Checking whether this statistics is valid.
	 * @return true if this statistics is valid.
	 */
	public boolean checkValid() {
		if (!Util.isUsed(zStatistic))
			return false;
		
		if (xStatistic == null || xStatistic.length == 0)
			return false;
		for (double x : xStatistic) {
			if (!Util.isUsed(x))
				return false;
		}
		
		return true;
	}
	
	
}

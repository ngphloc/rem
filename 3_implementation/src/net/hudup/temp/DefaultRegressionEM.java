package net.hudup.temp;

import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.em.regression.RegressionEM;


/**
 * Expectation maximization algorithm for regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class DefaultRegressionEM extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of the number of independent variables, stored in configuration.
	 */
	protected final static String REM_REGRESSOR_NUMBER_FIELD = "rem_regressor_number";
	
	
	/**
	 * The default number of independent variables, stored in configuration.
	 */
	protected final static int REM_REGRESSOR_NUMBER_DEFAULT = 0;
	
	
	/**
	 * Variable contains complete data.
	 */
	protected List<double[]> data = Util.newList();
	
	
	/**
	 * Values of last column.
	 */
	protected double[] z = null;
	
	
	/**
	 * Attribute list for all variables: all xi, y, and z.
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public DefaultRegressionEM() {
		// TODO Auto-generated constructor stub
	}

	
	@Override
	public synchronized Object learn() throws Exception {
		// TODO Auto-generated method stub
		int n0 = getConfig().containsKey(REM_REGRESSOR_NUMBER_FIELD) ?
				getConfig().getAsInt(REM_REGRESSOR_NUMBER_FIELD) : 0;
		int n = 0;
		this.data.clear();
		this.z = null;
		List<Double> zValues = Util.newList();
		while (this.sample.next()) {
			Profile profile = sample.pick();
			if (profile.isMissing(profile.getAttCount() - 1))
				continue;
			double lastValue = profile.getValueAsReal(profile.getAttCount() - 1);
			if (!Util.isUsed(lastValue))
				continue;
			
			if (n <= 0) {
				if (n0 > 0)
					n = Math.min(profile.getAttCount() - 1, n0);
				else
					n = profile.getAttCount() - 1;
			}
			if (n <= 0) break;
			if (this.attList == null)
				this.attList = profile.getAttRef();
			
			boolean missing = false;
			double[] vector = new double[n + 1];
			vector[0] = 1;
			for (int j = 0; j < n; j++) {
				if (profile.isMissing(j)) {
					missing = true;
					break;
				}
				double value = profile.getValueAsReal(j);
				if (!Util.isUsed(value)) {
					missing = true;
					break;
				}
				vector[1 + j] = value;
			}
			
			if (missing) continue;
				
			this.data.add(vector);
			zValues.add(lastValue);
		}
		this.sample.close();
		
		if (this.data.size() == 0) {
			unsetup();
			return null;
		}
		
		this.z = new double[zValues.size()];
		for (int i = 0; i < this.z.length; i++)
			this.z[i] = zValues.get(i);
		zValues.clear();
		
		return super.learn();
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		this.data.clear();
		this.z = null;
		this.attList = null;
	}


	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		if (this.data.size() == 0)
			return null;
		
		int m = this.data.size();
		int size = this.data.get(0).length; // size = n + 1
		double[] parameter = (double[])currentParameter;
		for (int i = 0; i < m; i++) {
			double[] vector = this.data.get(i);
			double sum = 0;
			for (int j = 0; j < size; j++)
				sum += parameter[j] * vector[j];
			
			this.z[i] = sum;
		}
		return z;
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		if (currentStatistic == null || this.data.size() == 0)
			return null;
		RealVector z = new ArrayRealVector((double[])currentStatistic);
		RealMatrix X = MatrixUtils.createRealMatrix(this.data.toArray(new double[][] {}));
		RealMatrix Xt = X.transpose();
		RealVector vector = MatrixUtils.inverse(Xt.multiply(X)).multiply(Xt).operate(z);
		
		double[] parameter = new double[vector.getDimension()];
		for (int j = 0; j < vector.getDimension(); j++)
			parameter[j] = vector.getEntry(j);
		return parameter;
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		if (this.data.size() == 0)
			return null;
		
		RealVector z = new ArrayRealVector(this.z);
		RealMatrix X = MatrixUtils.createRealMatrix(this.data.toArray(new double[][] {}));
		RealMatrix Xt = X.transpose();
		RealVector vector = MatrixUtils.inverse(Xt.multiply(X)).multiply(Xt).operate(z);
		
		double[] parameter = new double[vector.getDimension()];
		for (int j = 0; j < vector.getDimension(); j++)
			parameter[j] = vector.getEntry(j);
		
		return parameter;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		double[] parameter1 = (double[])currentParameter;
		double[] parameter2 = (double[])estimatedParameter;
		double maxBias = 0;
		for (int i = 0; i < parameter1.length; i++) {
			double bias = Math.abs(parameter1[i] - parameter2[i]);
			if (maxBias < bias)
				maxBias = bias;
		}
		
		return maxBias <= getConfig().getAsReal(EM_EPSILON_FIELD);
	}

	
	@Override
	public synchronized Object execute(Object input) {
		double[] coeffs = (double[])estimatedParameter;
		if (coeffs == null || coeffs.length == 0)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null;
		Profile profile = (Profile)input;
		
		double sum = coeffs[0];
		int n = coeffs.length - 1;
		for (int j = 0; j < n; j++) {
			double value = profile.getValueAsReal(j);
			sum += coeffs[j + 1] * value; 
		}
		
		return sum;
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem.default";
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
		config.put(REM_REGRESSOR_NUMBER_FIELD, REM_REGRESSOR_NUMBER_DEFAULT); // The number of independent variables
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}
	
	
	@Override
	public String parameterToShownText(Object parameter) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof double[]))
			return "";
		double[] array = (double[])parameter;
		return TextParserUtil.toText(array, ",");
	}


	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		double[] coeffs = (double[])estimatedParameter;
		if (coeffs == null || coeffs.length == 0)
			return "";

		StringBuffer buffer = new StringBuffer();
		buffer.append(attList.get(attList.size() - 1).getName() + " = " + coeffs[0]);
		for (int i = 0; i < coeffs.length - 1; i++) {
			double coeff = coeffs[i + 1];
			if (coeff < 0)
				buffer.append(" - " + Math.abs(coeff) + "*" + attList.get(i).getName());
			else
				buffer.append(" + " + coeff + "*" + attList.get(i).getName());
		}
		
		buffer.append(", ");
		
		return buffer.toString();
	}

	
}

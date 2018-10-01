package net.hudup.regression;

import java.util.List;

import net.hudup.core.data.Attribute;

/**
 * This class represents the wrapper of a variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 * 
 */
public class VarWrapper {
	
	
	/**
	 * Variable index.
	 */
	protected int index = -1;

	
	/**
	 * Variable name.
	 */
	protected String name = null;
	
	
	/**
	 * Variable expression.
	 */
	protected String expr = null;
	
	
	/**
	 * Variable attribute.
	 */
	protected Attribute att = null;
	
	
	/**
	 * Tag information.
	 */
	protected int tag = 0;
	
	
	/**
	 * Constructor with specified name and index.
	 * @param index specified index.
	 * @param name specified name.
	 * @param expr specified expression.
	 * @param att specified attribute.
	 */
	private VarWrapper(int index, String name, String expr, Attribute att) {
		this.index = index;
		this.name = name;
		this.expr = expr;
		this.att = att;
	}
	
	
	/**
	 * Create by name.
	 * @param index specified index.
	 * @param name specified name.
	 * @param att specified attribute.
	 * @return variable created.
	 */
	public static VarWrapper createByName(int index, String name, Attribute att) {
		return new VarWrapper(index, name, null, att);
	}
	
	
	/**
	 * Create by expression.
	 * @param index specified index.
	 * @param expr specified expression.
	 * @param att specified attribute.
	 * @return variable created.
	 */
	public static VarWrapper createByExpr(int index, String expr, Attribute att) {
		return new VarWrapper(index, null, expr, att);
	}

	
	/**
	 * Getting index.
	 * @return index.
	 */
	public int getIndex() {
		return index;
	}

	
	/**
	 * Getting name.
	 * @return name.
	 */
	public String getName() {
		return name;
	}
	
	
	/**
	 * Getting variable expression.
	 * @return variable expression.
	 */
	public String getExpr() {
		return expr;
	}
	
	
	/**
	 * Getting variable attribute.
	 * @return
	 */
	public Attribute getAttribute() {
		return att;
	}
	
	
	/**
	 * Getting tag information.
	 * @return tag information.
	 */
	public int getTag() {
		return tag;
	}
	
	
	/**
	 * Setting tag information.
	 * @param tag specified tag information.
	 */
	public void setTag(int tag) {
		this.tag = tag;
	}
	
	
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		if (name != null)
			return name;
		else
			return expr;
	}
	
	
	/**
	 * Looking an variable by specified expression or name in specified variable list.
	 * @param varList specified variable list.
	 * @param lookupText specified expression or name.
	 * @return index of found variable in specified variable list.
	 */
	public static int lookup(List<VarWrapper> varList, String lookupText) {
		for (int i = 0; i < varList.size(); i++) {
			if (varList.get(i).toString().equals(lookupText))
				return i;
		}
		return -1;
	}
	
	
}

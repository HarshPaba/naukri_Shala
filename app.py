# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from distutils.log import debug 
from fileinput import filename 
from flask import Flask,request, render_template
import os
import job
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/') 
def main(): 
	return render_template("index.html") 

@app.route('/success', methods = ['POST']) 
def success(): 
	if request.method == 'POST': 
		f = request.files['file'] 
		f.save("resume.pdf") 
		df1=job.builder("resume.pdf")
		os.remove("resume.pdf")
		return render_template("simple.html", column_names=df1.columns.values, row_data=list(df1.values.tolist()),
                           link_column="url", zip=zip)
# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(debug=True)







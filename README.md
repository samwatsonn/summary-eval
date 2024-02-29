# CommonLit - Summary Evaluation
For COMP6237: Data Mining - Coursework 1

By Samuel Watson, Joseph Padden, Daniel Turato, and Ritam Behwal.

## Setup

Clone the repo to your local machine using:
```angular2html
git clone https://github.com/samwatsonn/summary_eval.git
```

Navigate to the project folder
```angular2html
cd summary_eval
```

Create a virtual environment
```
python -m venv venv
```

Activate the environment
```
// Windows
./venv/Scripts/activate

// Linux / Mac OS
source venv/bin/activate
```

Now install the packages
```
pip install -r requirements.txt
```

You should be good to go!

## How to add new packages
To install a package into your **local** environment
```
pip install package_name
```
To update the requirements.txt
```
pip freeze > requirements.txt
```
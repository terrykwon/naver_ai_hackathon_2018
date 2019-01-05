# Naver AI Hackathon 2018

* https://github.com/AiHackathon2018/AI-Vision
* https://hack.nsml.navercorp.com/


## 1. 폴더 구조

* 실제 `nsml_run` 할 코드는 `round_1/main/` 안에 있습니다.
* `notebooks`에는 그냥 실험용 jupyter notebook 모음입니다.
* `data`안에 있는 파일들을 샘플입니다.


## 2. 설치 및 실행

* nsml binary를 https://hack.nsml.navercorp.com/ 에서 받은 후, `$PATH`에 포함되도록 `/usr/local/bin/`같은 곳에 옮깁니다.
* `cd round_1/main/` 후 `nsml run -d ir_ph1_v2 -e main.py` 를 하면 nsml 서버에서 `main.py`가 실행될겁니다.


## 3. 협업 방법

* 본인 이름으로 새로운 branch를 만듭니다. e.g.) `git checkout -b terry`
* 거기서 마음대로 작업합니다.
* **절대 `master`과 merge 하면 안됩니다.**

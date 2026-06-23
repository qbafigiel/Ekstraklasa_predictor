import pandas as pd
import time
import re
from playwright.sync_api import sync_playwright

PLIK_WYNIK = "data/podania_uzup_2025_26.csv"

MECZE = [
    ("C4whHmi3", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=C4whHmi3"),
    ("bqKn6Nr2", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=bqKn6Nr2"),
    ("6ou8D5jS", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=6ou8D5jS"),
    ("vwIF9dkf", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=vwIF9dkf"),
    ("Ei2ZTQz9", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=Ei2ZTQz9"),
    ("YJR0FRLF", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=YJR0FRLF"),
    ("QVLv8qEk", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=QVLv8qEk"),
    ("INT6BIKs", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=INT6BIKs"),
    ("n3LN7zK6", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=n3LN7zK6"),
    ("nsSrjBh4", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=nsSrjBh4"),
    ("vcthnTiT", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=vcthnTiT"),
    ("lWice6Uq", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=lWice6Uq"),
    ("pbo6gpad", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=pbo6gpad"),
    ("b1DI2lxA", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=b1DI2lxA"),
    ("KfQjlkNG", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=KfQjlkNG"),
    ("AuXNt7Fj", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=AuXNt7Fj"),
    ("h06R0S6M", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=h06R0S6M"),
    ("jyAA4A7c", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=jyAA4A7c"),
    ("lAbIws3s", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=lAbIws3s"),
    ("rZlKQZ7e", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=rZlKQZ7e"),
    ("Imegufvn", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=Imegufvn"),
    ("IkjZZb26", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=IkjZZb26"),
    ("dz2QyLXg", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=dz2QyLXg"),
    ("O8tlUTVp", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=O8tlUTVp"),
    ("46lsYxXI", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=46lsYxXI"),
    ("vy5HZk8N", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=vy5HZk8N"),
    ("jFYZiXNi", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=jFYZiXNi"),
    ("j1g1wY8b", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=j1g1wY8b"),
    ("nTnAyCwB", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=nTnAyCwB"),
    ("z7YrPsll", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=z7YrPsll"),
    ("rXyiNLI0", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=rXyiNLI0"),
    ("jZdCU3QQ", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=jZdCU3QQ"),
    ("UPHoq3uK", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=UPHoq3uK"),
    ("r9g4WPeE", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=r9g4WPeE"),
    ("GnmSODx8", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=GnmSODx8"),
    ("xWUaLamD", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=xWUaLamD"),
    ("GO4uYHFL", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=GO4uYHFL"),
    ("UVOpCzg2", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=UVOpCzg2"),
    ("n7eBSewq", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=n7eBSewq"),
    ("EgF2ac9d", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=EgF2ac9d"),
    ("Y3VyEdOk", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=Y3VyEdOk"),
    ("fkhkN013", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=fkhkN013"),
    ("xYq6JI0S", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=xYq6JI0S"),
    ("f9GWZwo9", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=f9GWZwo9"),
    ("t6jcLvWF", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=t6jcLvWF"),
    ("nRJm5SuI", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=nRJm5SuI"),
    ("8KWLjTQO", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=8KWLjTQO"),
    ("6TBf1Jvp", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=6TBf1Jvp"),
    ("no2WAWBt", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=no2WAWBt"),
    ("vFeELmQb", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=vFeELmQb"),
    ("8t8d36AU", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=8t8d36AU"),
    ("b5m6N9en", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=b5m6N9en"),
    ("v1Gu7lB5", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=v1Gu7lB5"),
    ("CSgMJRfB", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=CSgMJRfB"),
    ("vuu5fBRa", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=vuu5fBRa"),
    ("YsxcdXdm", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=YsxcdXdm"),
    ("hA1OsYsf", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=hA1OsYsf"),
    ("EceQ1gZP", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/cracovia-KvXSf2A6/szczegoly/statystyki/?mid=EceQ1gZP"),
    ("WfDFqfDs", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=WfDFqfDs"),
    ("QyFVuCC6", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=QyFVuCC6"),
    ("0fTDhkdC", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=0fTDhkdC"),
    ("M38vvjsJ", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=M38vvjsJ"),
    ("0CMqkGkK", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=0CMqkGkK"),
    ("CWqYzQU9", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=CWqYzQU9"),
    ("4UOCTsNq", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=4UOCTsNq"),
    ("ChL8m3EF", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=ChL8m3EF"),
    ("jwXakPq3", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=jwXakPq3"),
    ("8bwsZ3aM", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=8bwsZ3aM"),
    ("p4vcRCcD", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=p4vcRCcD"),
    ("OWvERLid", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=OWvERLid"),
    ("CjtkTYS0", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=CjtkTYS0"),
    ("IimtVfbl", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=IimtVfbl"),
    ("h8lQ7R0N", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=h8lQ7R0N"),
    ("46tQxn0c", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=46tQxn0c"),
    ("8pb8K4pT", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=8pb8K4pT"),
    ("v56iOno4", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=v56iOno4"),
    ("proI9mWA", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=proI9mWA"),
    ("xtQd25Op", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=xtQd25Op"),
    ("CI4aMQFG", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/gks-katowice-K4AgRmS1/szczegoly/statystyki/?mid=CI4aMQFG"),
    ("rwDrQ8Gi", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=rwDrQ8Gi"),
    ("OA3wFr8j", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=OA3wFr8j"),
    ("f5tnDh2M", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=f5tnDh2M"),
    ("UBYpBujI", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=UBYpBujI"),
    ("0WtUENjg", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=0WtUENjg"),
    ("fJBTGqMt", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=fJBTGqMt"),
    ("8O69Blp1", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=8O69Blp1"),
    ("xxxg9JKU", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=xxxg9JKU"),
    ("G2g1DVWo", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=G2g1DVWo"),
    ("lGi9B91b", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=lGi9B91b"),
    ("8dpxD1L5", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=8dpxD1L5"),
    ("4j0I9SED", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=4j0I9SED"),
    ("WITqnYGk", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=WITqnYGk"),
    ("SpNhpCo2", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=SpNhpCo2"),
    ("fLtzckV7", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=fLtzckV7"),
    ("QukJ0XVr", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=QukJ0XVr"),
    ("G63m6g8L", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=G63m6g8L"),
    ("0b31DAFl", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=0b31DAFl"),
    ("ATOMKhNE", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=ATOMKhNE"),
    ("M7rSbB0e", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=M7rSbB0e"),
    ("fiFv8Fw9", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=fiFv8Fw9"),
    ("EoX6jYko", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=EoX6jYko"),
    ("GWnvuH9M", "https://www.flashscore.pl/mecz/pilka-nozna/wisla-plock-QoZYVU3E/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=GWnvuH9M"),
    ("fwz6gGOF", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=fwz6gGOF"),
    ("zmrEifgS", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=zmrEifgS"),
    ("nieNraAc", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=nieNraAc"),
    ("nDWcexf3", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=nDWcexf3"),
    ("rJlWtwuA", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=rJlWtwuA"),
    ("bij3ZfXq", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=bij3ZfXq"),
    ("Kp3yLeIj", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=Kp3yLeIj"),
    ("6HgBXY1d", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=6HgBXY1d"),
    ("hCODSnKa", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=hCODSnKa"),
    ("ngQTO4JO", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=ngQTO4JO"),
    ("bXgnsnZh", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=bXgnsnZh"),
    ("Q1vZZURP", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/gks-katowice-K4AgRmS1/szczegoly/statystyki/?mid=Q1vZZURP"),
    ("pnL5U8km", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=pnL5U8km"),
    ("ERCMQQlC", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=ERCMQQlC"),
    ("QJevq84t", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=QJevq84t"),
    ("0dj3w4YH", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=0dj3w4YH"),
    ("fameuQ35", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=fameuQ35"),
    ("WMJe3Wjl", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=WMJe3Wjl"),
    ("Y7imCRZI", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=Y7imCRZI"),
    ("AT6NHV5s", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=AT6NHV5s"),
    ("SlD41AL0", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=SlD41AL0"),
    ("0vTP8hTQ", "https://www.flashscore.pl/mecz/pilka-nozna/widzew-lodz-rNOIW3uC/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=0vTP8hTQ"),
    ("02BCaljD", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=02BCaljD"),
    ("WreWF9zf", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=WreWF9zf"),
    ("v9bvEm56", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=v9bvEm56"),
    ("voGiNirK", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=voGiNirK"),
    ("vPvHAEbE", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=vPvHAEbE"),
    ("betFjg7r", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=betFjg7r"),
    ("W8vNlXye", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=W8vNlXye"),
    ("IwQYANhc", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=IwQYANhc"),
    ("zF0l3BZj", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=zF0l3BZj"),
    ("SzYVnB67", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=SzYVnB67"),
    ("84G9AgsH", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=84G9AgsH"),
    ("4W9I8XBT", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=4W9I8XBT"),
    ("4fOt91N9", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=4fOt91N9"),
    ("6XZjSZVg", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=6XZjSZVg"),
    ("4dg9seDU", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=4dg9seDU"),
    ("rNOJ1bbn", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=rNOJ1bbn"),
    ("dWk1qHrI", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=dWk1qHrI"),
    ("GzNRaxTb", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=GzNRaxTb"),
    ("thPhEyrh", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=thPhEyrh"),
    ("ziI1CFC4", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=ziI1CFC4"),
    ("UDezdfSN", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=UDezdfSN"),
    ("MmhScGcB", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=MmhScGcB"),
    ("xtphkLM7", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=xtphkLM7"),
    ("UkFcN5Yj", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=UkFcN5Yj"),
    ("lYpDZHwK", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=lYpDZHwK"),
    ("MDyJWwaC", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=MDyJWwaC"),
    ("Yi5ykLEt", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=Yi5ykLEt"),
    ("IuVRUHTO", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=IuVRUHTO"),
    ("hzurVQN7", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=hzurVQN7"),
    ("UJbpmaqg", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=UJbpmaqg"),
    ("zH1howE5", "https://www.flashscore.pl/mecz/pilka-nozna/widzew-lodz-rNOIW3uC/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=zH1howE5"),
    ("Esk2zMam", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=Esk2zMam"),
    ("KMtzXnge", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=KMtzXnge"),
    ("8rNUhUFE", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=8rNUhUFE"),
    ("jHJrVdmR", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=jHJrVdmR"),
    ("4z0UPVoS", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=4z0UPVoS"),
    ("0r3DTDn3", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=0r3DTDn3"),
    ("06o0K99k", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=06o0K99k"),
    ("KnQQ5Afd", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=KnQQ5Afd"),
    ("8UsdvV0L", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=8UsdvV0L"),
    ("nFgmtiW8", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=nFgmtiW8"),
    ("886LRiHF", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=886LRiHF"),
    ("K811Imv2", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=K811Imv2"),
    ("vDYRY8Or", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=vDYRY8Or"),
    ("lpBDbgmT", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=lpBDbgmT"),
    ("EkNI7WPq", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=EkNI7WPq"),
    ("QPlVGz3c", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=QPlVGz3c"),
    ("phEd2yl4", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=phEd2yl4"),
    ("MirvFEX9", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=MirvFEX9"),
    ("YHnpgXAj", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=YHnpgXAj"),
    ("tK750FIG", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=tK750FIG"),
    ("CvGl4cJi", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=CvGl4cJi"),
    ("Obwwj7bs", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=Obwwj7bs"),
    ("WpIL0OCa", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=WpIL0OCa"),
    ("h06Ub2sC", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=h06Ub2sC"),
    ("pERoloTg", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=pERoloTg"),
    ("baY1tfRp", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=baY1tfRp"),
    ("l6O2p3SI", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=l6O2p3SI"),
    ("hvVgnPc6", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=hvVgnPc6"),
    ("t0Bl86KP", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=t0Bl86KP"),
    ("OKED2prm", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=OKED2prm"),
    ("jTySbsQE", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=jTySbsQE"),
    ("SfjXx03e", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=SfjXx03e"),
    ("xQBhprBL", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=xQBhprBL"),
    ("8YX7eTzK", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=8YX7eTzK"),
    ("MHB6iaIl", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=MHB6iaIl"),
    ("CrQzcLfR", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=CrQzcLfR"),
    ("OMqtyvY7", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=OMqtyvY7"),
    ("Iq5Fkwm1", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=Iq5Fkwm1"),
    ("fshPvMYr", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=fshPvMYr"),
    ("xhYPrVZG", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=xhYPrVZG"),
    ("t6lNO9ro", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=t6lNO9ro"),
    ("4vzIpi54", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=4vzIpi54"),
    ("Ua8pn4t9", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=Ua8pn4t9"),
    ("l4VcO0l2", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=l4VcO0l2"),
    ("tvSkQMJk", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=tvSkQMJk"),
    ("jyArWijB", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=jyArWijB"),
    ("8Ct9nDzh", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=8Ct9nDzh"),
    ("dYCjUVKN", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=dYCjUVKN"),
    ("zoLR4FiC", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=zoLR4FiC"),
    ("W63qNd7t", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=W63qNd7t"),
    ("dEl9HhyI", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=dEl9HhyI"),
    ("MeIRZZjn", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=MeIRZZjn"),
    ("za00JE65", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=za00JE65"),
    ("rVnHFW5U", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=rVnHFW5U"),
    ("SYdhLzxg", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=SYdhLzxg"),
    ("f9GZXDLb", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=f9GZXDLb"),
    ("GfTZ2gMO", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=GfTZ2gMO"),
    ("Sbtqfbwf", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=Sbtqfbwf"),
    ("zNxuEJUP", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=zNxuEJUP"),
    ("65TA8chm", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=65TA8chm"),
    ("SKHJ6yNa", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=SKHJ6yNa"),
    ("z7YGjPW1", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/cracovia-KvXSf2A6/szczegoly/statystyki/?mid=z7YGjPW1"),
    ("OQZihx86", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=OQZihx86"),
    ("0pQ2ESO8", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=0pQ2ESO8"),
    ("dvYajGxJ", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=dvYajGxJ"),
    ("KYxydK8s", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=KYxydK8s"),
    ("YeRnmjQ9", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=YeRnmjQ9"),
    ("69PfoUfM", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=69PfoUfM"),
    ("vcYwkCec", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=vcYwkCec"),
    ("EFnM1ku3", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=EFnM1ku3"),
    ("vuu8ho2k", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=vuu8ho2k"),
    ("YJ9Zwnmd", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=YJ9Zwnmd"),
    ("ns0TaT9F", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=ns0TaT9F"),
    ("Qg6Ru8Iq", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=Qg6Ru8Iq"),
    ("CjthR83j", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=CjthR83j"),
    ("d2WPm8bU", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=d2WPm8bU"),
    ("G4OWqacd", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=G4OWqacd"),
    ("vJRmzqVN", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=vJRmzqVN"),
    ("Sxm4xZBi", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=Sxm4xZBi"),
    ("I3wux50B", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=I3wux50B"),
    ("Kh1DYjBG", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=Kh1DYjBG"),
    ("QeLXwRVb", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=QeLXwRVb"),
    ("UTROum1n", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/korona-kielce-pp78XcbA/szczegoly/statystyki/?mid=UTROum1n"),
    ("A1IJUMho", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=A1IJUMho"),
    ("SI4LWUuT", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=SI4LWUuT"),
    ("0joCzDt4", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/gks-katowice-K4AgRmS1/szczegoly/statystyki/?mid=0joCzDt4"),
    ("dIOdxEff", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=dIOdxEff"),
    ("v1GDYVgJ", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=v1GDYVgJ"),
    ("MaAEPzBl", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=MaAEPzBl"),
    ("lC1VLhAD", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=lC1VLhAD"),
    ("rZM4zhP6", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=rZM4zhP6"),
    ("4KqIqd2K", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=4KqIqd2K"),
    ("z3zlvzQs", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=z3zlvzQs"),
    ("Gb3NNEu1", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=Gb3NNEu1"),
    ("OU9uDcRr", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=OU9uDcRr"),
    ("4pT8IJ3L", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=4pT8IJ3L"),
    ("U1VZWIIE", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=U1VZWIIE"),
    ("2ewl9FQ7", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=2ewl9FQ7"),
    ("dKP0KuZ8", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=dKP0KuZ8"),
    ("GOnEdxs2", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=GOnEdxs2"),
    ("ARq6bbCk", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=ARq6bbCk"),
    ("dr8mByde", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=dr8mByde"),
    ("js3Z3KlS", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=js3Z3KlS"),
    ("lxTHkUTH", "https://www.flashscore.pl/mecz/pilka-nozna/wisla-plock-QoZYVU3E/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=lxTHkUTH"),
    ("KAGZ7Bqn", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=KAGZ7Bqn"),
    ("M1BI72k3", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=M1BI72k3"),
    ("hOEs6kEb", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/gornik-zabrze-2LH3Ywq4/szczegoly/statystyki/?mid=hOEs6kEb"),
    ("fkzOoLSq", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=fkzOoLSq"),
    ("IT4R5tKF", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=IT4R5tKF"),
    ("lj8j4TrB", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=lj8j4TrB"),
    ("jL9fneNP", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=jL9fneNP"),
    ("zDR7OiVI", "https://www.flashscore.pl/mecz/pilka-nozna/legia-warszawa-K6kUepBs/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=zDR7OiVI"),
    ("projyVEO", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=projyVEO"),
    ("lGiswipC", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=lGiswipC"),
    ("SMv9ija5", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=SMv9ija5"),
    ("Kdy1gCUh", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/gks-katowice-K4AgRmS1/szczegoly/statystyki/?mid=Kdy1gCUh"),
    ("Kz0QtZpm", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=Kz0QtZpm"),
    ("0fLaeYat", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=0fLaeYat"),
    ("tjlZvDFa", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=tjlZvDFa"),
    ("hn1AFdnl", "https://www.flashscore.pl/mecz/pilka-nozna/lechia-gdansk-GGLmkiK8/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=hn1AFdnl"),
    ("0EqxY42F", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=0EqxY42F"),
    ("Y5MqUe1s", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/jagiellonia-bialystok-lIDaZJTc/szczegoly/statystyki/?mid=Y5MqUe1s"),
    ("2aTaQD06", "https://www.flashscore.pl/mecz/pilka-nozna/widzew-lodz-rNOIW3uC/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=2aTaQD06"),
    ("hltUZQX2", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=hltUZQX2"),
    ("QaaIDzH0", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=QaaIDzH0"),
    ("hO0hqmeA", "https://www.flashscore.pl/mecz/pilka-nozna/radomiak-radom-zD5nYhAT/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=hO0hqmeA"),
    ("Qya0sRPM", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=Qya0sRPM"),
    ("8d2Py5mp", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=8d2Py5mp"),
    ("GM5XZrId", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=GM5XZrId"),
    ("hALJUmtH", "https://www.flashscore.pl/mecz/pilka-nozna/nieciecza-YNkK4khO/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=hALJUmtH"),
    ("drSAW9B4", "https://www.flashscore.pl/mecz/pilka-nozna/piast-gliwice-ve2oT9ck/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=drSAW9B4"),
    ("dd7qo9Qc", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=dd7qo9Qc"),
    ("xYByIAdB", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=xYByIAdB"),
    ("YuFKLhcn", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=YuFKLhcn"),
    ("8MoGASlo", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/widzew-lodz-rNOIW3uC/szczegoly/statystyki/?mid=8MoGASlo"),
    ("bm0pGlRN", "https://www.flashscore.pl/mecz/pilka-nozna/rakow-czestochowa-SQOrbYim/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=bm0pGlRN"),
    ("OUo23BsI", "https://www.flashscore.pl/mecz/pilka-nozna/lech-poznan-OpNH7Ouf/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=OUo23BsI"),
    ("ITvB1kCU", "https://www.flashscore.pl/mecz/pilka-nozna/motor-lublin-IoLk2VlL/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=ITvB1kCU"),
    ("OUT2YVth", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/piast-gliwice-ve2oT9ck/szczegoly/statystyki/?mid=OUT2YVth"),
    ("G4rf5XC5", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=G4rf5XC5"),
    ("U78TJWRb", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=U78TJWRb"),
    ("jqAR3Ky1", "https://www.flashscore.pl/mecz/pilka-nozna/pogon-szczecin-Um9YwPQ0/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=jqAR3Ky1"),
    ("2JyvDOEE", "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/motor-lublin-IoLk2VlL/szczegoly/statystyki/?mid=2JyvDOEE"),
    ("fHGI5t7l", "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/?mid=fHGI5t7l"),
    ("Od8Z1b6D", "https://www.flashscore.pl/mecz/pilka-nozna/gks-katowice-K4AgRmS1/rakow-czestochowa-SQOrbYim/szczegoly/statystyki/?mid=Od8Z1b6D"),
    ("tEjqishe", "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/wisla-plock-QoZYVU3E/szczegoly/statystyki/?mid=tEjqishe"),
    ("OYPnB2qR", "https://www.flashscore.pl/mecz/pilka-nozna/widzew-lodz-rNOIW3uC/zaglebie-lubin-tlYOere0/szczegoly/statystyki/?mid=OYPnB2qR"),
    ("nPjw9FDt", "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/lech-poznan-OpNH7Ouf/szczegoly/statystyki/?mid=nPjw9FDt"),
    ("fkfo7grg", "https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/nieciecza-YNkK4khO/szczegoly/statystyki/?mid=fkfo7grg"),]

MAPA = {
    "Podania":                              ("podania_sk_gosp",       "podania_wszy_gosp",
                                             "podania_sk_gosc",       "podania_wszy_gosc"),
    "Długie podania":                       ("dl_pod_sk_gosp",        "dl_pod_wszy_gosp",
                                             "dl_pod_sk_gosc",        "dl_pod_wszy_gosc"),
    "Dośrodkowania":                        ("dosrod_sk_gosp",        "dosrod_wszy_gosp",
                                             "dosrod_sk_gosc",        "dosrod_wszy_gosc"),
    "Próby odbioru piłki":                  ("odbiory_sk_gosp",       "odbiory_wszy_gosp",
                                             "odbiory_sk_gosc",       "odbiory_wszy_gosc"),
    "Podania w strefę obrony przeciwnika":  ("pod_strefa_sk_gosp",    "pod_strefa_wszy_gosp",
                                             "pod_strefa_sk_gosc",    "pod_strefa_wszy_gosc"),
}


def znajdz_nawias_wstecz(linie, od):
    for j in range(od - 1, max(0, od - 5), -1):
        m = re.search(r'\((\d+)/(\d+)\)', linie[j].strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def znajdz_nawias_wprzod(linie, od):
    for j in range(od + 1, min(len(linie), od + 5)):
        m = re.search(r'\((\d+)/(\d+)\)', linie[j].strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def parsuj(linie):
    wyniki = {}
    przetworzone = set()
    for i, linia in enumerate(linie):
        nazwa = linia.strip()
        if nazwa not in MAPA or nazwa in przetworzone:
            continue
        sk_g, ws_g, sk_gc, ws_gc = MAPA[nazwa]
        sk_gosp, ws_gosp = znajdz_nawias_wstecz(linie, i)
        sk_gosc, ws_gosc = znajdz_nawias_wprzod(linie, i)
        if sk_gosp is not None:
            wyniki[sk_g] = sk_gosp
            wyniki[ws_g] = ws_gosp
        if sk_gosc is not None:
            wyniki[sk_gc] = sk_gosc
            wyniki[ws_gc] = ws_gosc
        przetworzone.add(nazwa)
    return wyniki


if __name__ == "__main__":
    # Wznowienie
    try:
        df_juz = pd.read_csv(PLIK_WYNIK)
        juz_pobrane = set(df_juz["flash_id"].dropna().tolist())
        print(f"Wznowienie — już pobrane: {len(juz_pobrane)}")
    except FileNotFoundError:
        df_juz = pd.DataFrame()
        juz_pobrane = set()

    wyniki = []
    restart_co = 40

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9"
        })

        for nr, (flash_id, url) in enumerate(MECZE):
            if flash_id in juz_pobrane:
                continue

            print(f"[{nr+1}/{len(MECZE)}] {flash_id}", end=" ", flush=True)

            if nr > 0 and nr % restart_co == 0:
                print("\n  --- Restart browsera ---")
                try: page.close(); browser.close()
                except: pass
                time.sleep(8)
                browser = p.firefox.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "pl-PL,pl;q=0.9"
                })

            sukces = False
            for attempt in range(4):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=25000)
                    page.wait_for_timeout(2500)
                    try:
                        page.click("button#onetrust-accept-btn-handler", timeout=2000)
                        page.wait_for_timeout(400)
                    except: pass
                    tekst = page.inner_text("body")
                    linie = [l.strip() for l in tekst.split("\n") if l.strip()]
                    if "Podania" in linie:
                        stat = parsuj(linie)
                        stat["flash_id"] = flash_id
                        stat["url"] = url
                        wyniki.append(stat)
                        pod = f"{stat.get('podania_sk_gosp','?')}/{stat.get('podania_wszy_gosp','?')}"
                        dos = f"{stat.get('dosrod_sk_gosp','?')}/{stat.get('dosrod_wszy_gosp','?')}"
                        print(f"podania:{pod} dosrod:{dos}")
                        sukces = True
                        break
                    else:
                        print(f" brak(p{attempt+1})", end="", flush=True)
                        time.sleep(8 * (attempt + 1))
                except Exception as e:
                    print(f" err(p{attempt+1})", end="", flush=True)
                    try: page.close()
                    except: pass
                    page = browser.new_page()
                    page.set_extra_http_headers({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept-Language": "pl-PL,pl;q=0.9"
                    })
                    time.sleep(10 * (attempt + 1))

            if not sukces:
                print(" POMINIĘTO")
                wyniki.append({"flash_id": flash_id, "url": url})

            if len(wyniki) % 30 == 0 and len(wyniki) > 0:
                df_tmp = pd.concat([df_juz, pd.DataFrame(wyniki)], ignore_index=True)
                df_tmp.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
                print(f"  --- Checkpoint {len(df_tmp)} ---")

            time.sleep(1.5)

        try: page.close(); browser.close()
        except: pass

    # Ręczne mecze
    reczne = [
        {"flash_id": "MRhRBEoD", "url": "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=MRhRBEoD",
         "podania_sk_gosp": 248, "podania_wszy_gosp": 315, "podania_sk_gosc": 355, "podania_wszy_gosc": 428,
         "dl_pod_sk_gosp": 18, "dl_pod_wszy_gosp": 58, "dl_pod_sk_gosc": 27, "dl_pod_wszy_gosc": 53,
         "dosrod_sk_gosp": 4, "dosrod_wszy_gosp": 17, "dosrod_sk_gosc": 4, "dosrod_wszy_gosc": 15,
         "odbiory_sk_gosp": 10, "odbiory_wszy_gosp": 15, "odbiory_sk_gosc": 9, "odbiory_wszy_gosc": 19},
        {"flash_id": "pIwRSRAT", "url": "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=pIwRSRAT",
         "podania_sk_gosp": 388, "podania_wszy_gosp": 473, "podania_sk_gosc": 213, "podania_wszy_gosc": 284,
         "dl_pod_sk_gosp": 34, "dl_pod_wszy_gosp": 48, "dl_pod_sk_gosc": 22, "dl_pod_wszy_gosc": 54,
         "dosrod_sk_gosp": 11, "dosrod_wszy_gosp": 37, "dosrod_sk_gosc": 6, "dosrod_wszy_gosc": 19,
         "odbiory_sk_gosp": 7, "odbiory_wszy_gosp": 9, "odbiory_sk_gosc": 8, "odbiory_wszy_gosc": 15},
    ]

    df_nowe = pd.DataFrame(wyniki)
    df_final = pd.concat([df_juz, df_nowe], ignore_index=True)
    for r in reczne:
        if r["flash_id"] not in set(df_final["flash_id"].dropna().tolist()):
            df_final = pd.concat([df_final, pd.DataFrame([r])], ignore_index=True)

    df_final.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
    print(f"\nZapisano {len(df_final)} wierszy do {PLIK_WYNIK}")
    ok = df_final["podania_wszy_gosp"].notna().sum() if "podania_wszy_gosp" in df_final.columns else 0
    print(f"Mają podania_wszy_gosp: {ok}/{len(df_final)}")
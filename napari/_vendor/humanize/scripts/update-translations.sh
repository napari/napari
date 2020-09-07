set -e

# extract new phrases
/usr/local/opt/gettext/bin/xgettext --from-code=UTF-8 -o humanize.pot -k'_' -k'N_' -k'P_:1c,2' -l python src/humanize/*.py

for d in src/humanize/locale/*/; do
    locale="$(basename $d)"
    echo "$locale"
    # add them to locale files
    /usr/local/opt/gettext/bin/msgmerge -U src/humanize/locale/$locale/LC_MESSAGES/humanize.po humanize.pot
    # compile to binary .mo
    /usr/local/opt/gettext/bin/msgfmt --check -o src/humanize/locale/$locale/LC_MESSAGES/humanize{.mo,.po}
done

pybabel extract -F transparentai/babel.cfg -o transparentai/messages.pot .
# pybabel init -i transparentai/messages.pot -d transparentai/app/translations -l fr
pybabel update -i transparentai/messages.pot -d transparentai/app/translations
pybabel compile -d transparentai/app/translations
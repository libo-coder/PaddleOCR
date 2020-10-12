## txt字典说明
* `ppocr_keys_v1.txt` 是一个包含 6623 个字符的中文字典
* `ic15_dict.txt` 是一个包含 36 个字符的英文字典

## 自定义字典
如需要使用自定义的字典文件，需要在 `configs/rec/rec_icdar15_trian.yml` 中添加 `character_dict_path` 字段。指向自定义的字典路径，并将 `character_type` 设置为 `ch`.

## 添加空格类别
如果希望支持识别 “空格” 类别，需要将 yml 文件中的 use_space_char 字段设置为 true
**注意：** `use_space_char` 仅在 `character_type=ch` 时生效
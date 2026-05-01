# EventMemory LoCoMo real-timestamp ingest

## Parser

```
datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)
```

## Per-conversation summary

| Conversation | n_events | n_sessions | first_session | last_session | ingest_s |
| --- | --- | --- | --- | --- | --- |
| locomo_conv-26 | 419 | 19 | 2023-05-08T13:56:00+00:00 | 2023-10-22T09:55:00+00:00 | 2.66 |
| locomo_conv-30 | 369 | 19 | 2023-01-20T16:04:00+00:00 | 2023-07-23T18:46:00+00:00 | 1.91 |
| locomo_conv-41 | 663 | 32 | 2022-12-17T11:01:00+00:00 | 2023-08-16T11:08:00+00:00 | 2.8 |

## Parse samples

### locomo_conv-26

| turn_id | session_idx | session_date_time | parsed_iso | speaker | text |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1:56 pm on 8 May, 2023 | 2023-05-08T13:56:00+00:00 | Caroline | Hey Mel! Good to see you! How have you been? |
| 1 | 1 | 1:56 pm on 8 May, 2023 | 2023-05-08T13:56:00+00:00 | Melanie | Hey Caroline! Good to see you! I'm swamped with the kids & w |
| 2 | 1 | 1:56 pm on 8 May, 2023 | 2023-05-08T13:56:00+00:00 | Caroline | I went to a LGBTQ support group yesterday and it was so powe |
| 3 | 1 | 1:56 pm on 8 May, 2023 | 2023-05-08T13:56:00+00:00 | Melanie | Wow, that's cool, Caroline! What happened that was so awesom |
| 4 | 1 | 1:56 pm on 8 May, 2023 | 2023-05-08T13:56:00+00:00 | Caroline | The transgender stories were so inspiring! I was so happy an |

### locomo_conv-30

| turn_id | session_idx | session_date_time | parsed_iso | speaker | text |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 4:04 pm on 20 January, 2023 | 2023-01-20T16:04:00+00:00 | Gina | Hey Jon! Good to see you. What's up? Anything new? |
| 1 | 1 | 4:04 pm on 20 January, 2023 | 2023-01-20T16:04:00+00:00 | Jon | Hey Gina! Good to see you too. Lost my job as a banker yeste |
| 2 | 1 | 4:04 pm on 20 January, 2023 | 2023-01-20T16:04:00+00:00 | Gina | Sorry about your job Jon, but starting your own business sou |
| 3 | 1 | 4:04 pm on 20 January, 2023 | 2023-01-20T16:04:00+00:00 | Jon | Sorry to hear that! I'm starting a dance studio 'cause I'm p |
| 4 | 1 | 4:04 pm on 20 January, 2023 | 2023-01-20T16:04:00+00:00 | Gina | That's cool, Jon! What got you into this biz? |

### locomo_conv-41

| turn_id | session_idx | session_date_time | parsed_iso | speaker | text |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 11:01 am on 17 December, 2022 | 2022-12-17T11:01:00+00:00 | Maria | Hey John! Long time no see! What's up? |
| 1 | 1 | 11:01 am on 17 December, 2022 | 2022-12-17T11:01:00+00:00 | John | Hey Maria! Good to see you. Just got back from a family road |
| 2 | 1 | 11:01 am on 17 December, 2022 | 2022-12-17T11:01:00+00:00 | Maria | Been busy volunteering at the homeless shelter and keeping f |
| 3 | 1 | 11:01 am on 17 December, 2022 | 2022-12-17T11:01:00+00:00 | John | Woah, Maria, that sounds cool! I'm doing kickboxing and it's |
| 4 | 1 | 11:01 am on 17 December, 2022 | 2022-12-17T11:01:00+00:00 | Maria | Cool, John. Kickboxing is a perfect way to stay in shape and |

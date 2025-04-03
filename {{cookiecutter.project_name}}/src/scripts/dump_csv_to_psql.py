"""Data Dump script.

Populate the table "observations" of
our local postgres DB with the data from a csv file.

"""
import asyncio
from io import StringIO
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from skylight.utils.sql import load_session, get_connection
from dotenv import load_dotenv

import websockets
WEBSOCKET_URI="ws://localhost:1882/data_ingest"



project_path = Path(__file__).parents[2]
A = load_dotenv(os.path.join(project_path, ".env"))
if not A:
    raise Exception(".env not loaded.")


async def main(data_directory: str) -> None:
    """Dump the Sky-Light dataset onto timescale

    There are two types of files: logs and data files. 
    The datatype is used as a proxy for the uuid, since it
    is unclear from the original data files which column 
    names can be reliably used as a sensor ID.

    """

    # sql_connection = get_connection(
    #     user=os.environ["POSTGRES_USER"],
    #     password=os.environ["POSTGRES_PASSWORD"],
    #     host=os.environ["POSTGRES_HOST"],
    #     port=os.environ["POSTGRES_PORT"],
    #     database=os.environ["POSTGRES_DB"],
    # )

    #print(dir(sql_connection.connect().connection))


    async with websockets.connect(WEBSOCKET_URI) as websocket:
        n=1
        for (n, f) in enumerate(sort(os.listdir(data_directory))):
            filename = os.path.join(data_directory,f)

            if f.endswith(".csv"):
                try:
                    with open(filename,"r") as infile:
                        content = infile.read()
                        await websocket.send(content)
                        print(f"{n}")
                        n+=1
                except Exception as e:
                    print(f"Failure: {e}")





        # if n%10==0:
        #     print(f"At file {n}")

    
        # if "LOG_" in f:
        #     continue
        #     df = pd.read_csv(
        #         os.path.join(data_directory,f),
        #         delimiter=";",
        #         decimal=",",
        #         header=0,
        #         dtype={"Time": "string", "Id": np.dtype("int64"), "V3040_IV_MAIN_EXTRUDER_2DEC_B": "float"}
        #     )
        #     df_type = "log"
        # else:
        #     df = pd.read_csv(
        #         os.path.join(data_directory,f),
        #         header=0,
        #         delimiter=";",
        #         decimal=",",
        #         dtype={"Time": "string"}
        #         )

        #     df_type = "data"


        # df["uuid"]=df_type
        # df["data"]=df.drop(columns=["Time", "uuid"]).to_json(orient="records", lines=True).splitlines()
        # df["time"]=pd.to_datetime(df["Time"], format="%d-%m-%Y %H:%M:%S")

        # new_df = df[["time", "uuid", "data"]]

        # print(new_df)
        # #raise Exception
        # #output = StringIO()
        # new_df.to_csv("test.csv", quoting="", sep="\t", header=False, index=False, doublequote=False)
        # #output.seek(0)
        # #print(output)
        # raise Exception

        # with sql_connection.connect().connection.cursor() as cur:
        #     cur.copy_from(output, 'observations', sep='\t', columns=('time', 'uuid', 'data'))
        #     print(dir(cur))
        #     sql_connection.commit()
        
        # #new_df.to_sql("observations",index_label="Id", con=sql_connection, if_exists="append", index=False)

        # #insert_stmt = "INSERT INTO observations(time, uuid, data) VALUES (%s, %s, %s)"
        
        # # values = []

        # for row in df.itertuples(index=False):
        #     uuid = df_type
        #     time = row.Time
        #     data = json.dumps(row._asdict()).replace("NaN", "null")
        #     values.append(f"(TO_TIMESTAMP('{time}', 'dd-MM-yyyy HH24:MI:SS'), '{uuid}', '{data}')")

        # insert_stmt += ", ".join(values) + ";"



        # for i, row in df.iterrows():
        #     insert_stmt = "INSERT INTO observations(time, uuid, data) VALUES\n"
        #     uuid = df_type
        #     time = row["Time"]
        #     data = json.dumps(row.to_dict()).replace("NaN", "null")
        #     insert_stmt += f"(TO_TIMESTAMP('{time}', 'dd-MM-yyyy HH24:MI:SS'), '{uuid}', '{data}'),\n"
        #     insert_stmt = insert_stmt[:-2] + ";"


        # sql_connection.execute(text(insert_stmt))
        # sql_connection.commit()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("dump excel data to postgres")

    parser.add_argument(
        "-d",
        "--data-directory",
        help="Path to where the csv files are located",
        required=True,
    )

    args = parser.parse_args()

    asyncio.run(main(
        data_directory=args.data_directory,
    )
    )

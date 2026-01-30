CREATE DATABASE  IF NOT EXISTS vendas_dimensional;
USE vendas_dimensional;

DROP TABLE IF EXISTS dim_categoria;
CREATE TABLE dim_categoria (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  categoria varchar(50) NOT NULL UNIQUE
);

DROP TABLE IF EXISTS dim_cliente;
CREATE TABLE dim_cliente (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  nome varchar(200) NOT NULL,
  sexocodigo char(1),
  sexodescricao varchar(20),
  siglanacionalidade char(2) NOT NULL,
  nacionalidade varchar(200) NOT NULL,
  profissao varchar(200),
  email varchar(300) NOT NULL,
  provedordocliente varchar(300) NOT NULL,
  nascimento date NOT NULL,
  cadastro date NOT NULL,
  enderecocompleto varchar(1000) NOT NULL
);

DROP TABLE IF EXISTS dim_fornecedor;
CREATE TABLE dim_fornecedor (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  fornecedor varchar(200) NOT NULL,
  contato varchar(200)
);

DROP TABLE IF EXISTS dim_pais;
CREATE TABLE dim_pais (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  sigla char(2) NOT NULL UNIQUE,
  sigla3 char(3) NOT NULL UNIQUE,
  pais varchar(200) NOT NULL UNIQUE,
  nomeinternacional varchar(200) NOT NULL UNIQUE,
  continente varchar(200) NOT NULL,
  regiao varchar(200) NOT NULL
) ;

DROP TABLE IF EXISTS dim_pedido;
CREATE TABLE dim_pedido (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  numerodopedido bigint NOT NULL UNIQUE
);

DROP TABLE IF EXISTS dim_produto;
CREATE TABLE dim_produto (
  id bigint NOT NULL UNIQUE,
  idtransacional bigint NOT NULL UNIQUE,
  produto varchar(200) NOT NULL UNIQUE,
  precounitario double NOT NULL,
  descontinuado tinyint NOT NULL
);

DROP TABLE IF EXISTS dim_tempo;
CREATE TABLE dim_tempo (
  id bigint NOT NULL PRIMARY KEY,
  data date NOT NULL UNIQUE,
  dia smallint NOT NULL,
  mes smallint NOT NULL,
  ano smallint NOT NULL,
  data_juliana bigint NOT NULL,
  semestre smallint NOT NULL,
  quadrimestre smallint NOT NULL,
  trimestre smallint NOT NULL,
  bimestre smallint NOT NULL,
  nome_mes varchar(50),
  dia_da_semana smallint NOT NULL,
  nome_dia_da_semana varchar(50) NOT NULL,
  semana_do_ano smallint NOT NULL,
  data_string char(10) NOT NULL,
  dia_no_ano smallint NOT NULL,
  ultimo_dia_mes smallint NOT NULL,
  fim_de_semana tinyint NOT NULL
);

DROP TABLE IF EXISTS dim_transportadora;
CREATE TABLE dim_transportadora (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  transportadora varchar(200) NOT NULL,
  contato varchar(200) NOT NULL
);

DROP TABLE IF EXISTS dim_vendedor;
CREATE TABLE dim_vendedor (
  id bigint NOT NULL PRIMARY KEY,
  idtransacional bigint NOT NULL UNIQUE,
  nome varchar(200) NOT NULL,
  sexocodigo char(1) NOT NULL,
  nascimento date NOT NULL,
  contrato date NOT NULL,
  supervisor varchar(200),
  sexodescricao varchar(50) NOT NULL
);

DROP TABLE IF EXISTS fat_item;
CREATE TABLE fat_item (
  id bigint NOT NULL PRIMARY KEY,
  qtdvendida bigint NOT NULL,
  precounitarionavenda double NOT NULL,
  valorfrete double NOT NULL,
  valordesconto double NOT NULL,
  valorcomissao double NOT NULL,
  idtempopedido bigint NOT NULL,
  idtempopagamento bigint,
  idtempoprevisao bigint NOT NULL,
  idtempoenvio bigint,
  idtempoentrega bigint,
  idcliente bigint NOT NULL,
  idcategoria bigint NOT NULL,
  idproduto bigint NOT NULL,
  idvendedor bigint NOT NULL,
  idtransportadora bigint NOT NULL,
  idfornecedor bigint NOT NULL,
  idpedido bigint NOT NULL,
  idpaisorigemcliente bigint NOT NULL,
  idpaisnacionalidadecliente bigint NOT NULL,
  idpaisdestino bigint NOT NULL,
  idpaistransportadora bigint NOT NULL,
  idpaisvendedor bigint NOT NULL,
  idpaisfornecedor bigint NOT NULL
);

ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtempopedido) REFERENCES dim_tempo(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtempopagamento) REFERENCES dim_tempo(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtempoprevisao) REFERENCES dim_tempo(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtempoenvio) REFERENCES dim_tempo(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtempoentrega) REFERENCES dim_tempo(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idcliente) REFERENCES dim_cliente(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idcategoria) REFERENCES dim_categoria(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idproduto) REFERENCES dim_produto(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idvendedor) REFERENCES dim_vendedor(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idtransportadora) REFERENCES dim_transportadora(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpedido) REFERENCES dim_pedido(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idfornecedor) REFERENCES dim_fornecedor(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaisorigemcliente) REFERENCES dim_pais(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaisnacionalidadecliente) REFERENCES dim_pais(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaisdestino) REFERENCES dim_pais(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaistransportadora) REFERENCES dim_pais(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaisvendedor) REFERENCES dim_pais(id);
ALTER TABLE fat_item ADD CONSTRAINT FOREIGN KEY(idpaisfornecedor) REFERENCES dim_pais(id);